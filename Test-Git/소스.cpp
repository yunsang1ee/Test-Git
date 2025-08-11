#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxcapi.h>  // DXC ��� �߰�
#include <DirectXMath.h>
#include <wrl/client.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <atlbase.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxcompiler.lib")  // DXC ���̺귯��

using namespace DirectX;
using Microsoft::WRL::ComPtr;

// ���� ����ü
struct Vertex {
	XMFLOAT3 position;
	XMFLOAT3 normal;
};

struct SceneConstants {
	float time;
	float padding[3];
};

constexpr UINT kId = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;		// 32
constexpr UINT kRecord = ((kId + 31) & ~31);					// 32 ����
constexpr UINT kTable = ((kRecord + 63) & ~63);					// 64 ���� ����

// ����Ʈ���̽� ���̴� �ڵ� (������)
const char* g_raytracingShaderCode = R"(
RaytracingAccelerationStructure Scene : register(t0, space0);
RWTexture2D<float4> RenderTarget : register(u0);

struct RayPayload {
    float4 color;
    uint recursionDepth;
};

cbuffer SceneConstants : register(b0) {
    float g_time;
};

// �ݻ� ���� ���
float3 Reflect(float3 incident, float3 normal) {
    return incident - 2.0f * dot(incident, normal) * normal;
}

// HSV to RGB ���� ��ȯ (�� ������ �ִ� ������ ����)
float3 HSVtoRGB(float3 hsv) {
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(hsv.xxx + K.xyz) * 6.0 - K.www);
    return hsv.z * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), hsv.y);
}

[shader("raygeneration")]
void RaygenShader() {
    uint3 index = DispatchRaysIndex();
    uint3 dimensions = DispatchRaysDimensions();
    
    if (index.x >= dimensions.x || index.y >= dimensions.y) return;
    
    float2 pixelCenter = float2(index.xy) + 0.5f;
    float2 inUV = pixelCenter / float2(dimensions.xy);
    float2 d = inUV * 2.0f - 1.0f;
    
    // �ð� ��� ī�޶� �ִϸ��̼�
    float cameraRadius = 5.0f + 1.5f * sin(g_time * 0.4f);
    float cameraHeight = 2.0f + 1.0f * cos(g_time * 0.6f);
    
    float3 cameraPos = float3(
        cameraRadius * cos(g_time * 0.25f),
        cameraHeight,
        cameraRadius * sin(g_time * 0.25f)
    );
    
    float3 target = float3(0, 0, 0);
    float3 forward = normalize(target - cameraPos);
    float3 right = normalize(cross(forward, float3(0, 1, 0)));
    float3 up = cross(right, forward);
    
    // �þ߰� ����
    float fov = 50.0f * 3.14159265f / 180.0f;
    float aspectRatio = (float)dimensions.x / (float)dimensions.y;
    
    float3 rayDir = normalize(
        forward + 
        tan(fov * 0.5f) * (d.x * aspectRatio * right + d.y * up)
    );
    
    RayDesc ray;
    ray.Origin = cameraPos;
    ray.Direction = rayDir;
    ray.TMin = 0.001f;
    ray.TMax = 1000.0f;
    
    // �ʱ� ���̷ε� ����
    RayPayload payload;
    payload.color = float4(0, 0, 0, 1);
    payload.recursionDepth = 0;
    
    TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 1, 0, ray, payload);
    
    // ���� �� ���� (ACES ����� �ٻ�)
    float3 finalColor = payload.color.rgb;
    
    // ���� ����
    finalColor *= 1.2f;
    
    // ACES ����� �ٻ�
    float3 a = finalColor * 2.51f;
    float3 b = finalColor * 0.03f + 0.59f;
    float3 c = finalColor * 2.43f + 0.14f;
    finalColor = saturate((a) / (b + c));
    
    // ���� ����
    finalColor = pow(finalColor, float3(1.0f/2.2f, 1.0f/2.2f, 1.0f/2.2f));
    
    RenderTarget[index.xy] = float4(finalColor, 1.0f);
}

[shader("closesthit")]
void ClosestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    // �ִ� ��� ���� ����
    if (payload.recursionDepth >= 2) {
        payload.color = float4(0.05, 0.05, 0.1, 1);
        return;
    }
    
    uint primitiveIndex = PrimitiveIndex();
    uint faceIndex = primitiveIndex / 2;
        
    // �� �麰 ��� ����� ���� (�ξ� �� �Ƹ��ٿ� �����)
    float3 baseFaceColors[6];
    
    baseFaceColors[0] = float3(1.0f, 0.2f, 0.2f);	// �ո� - ������ ����
    baseFaceColors[1] = float3(0.2f, 1.0f, 0.2f);	// �޸� - ������ �ʷ�
    baseFaceColors[2] = float3(0.2f, 0.4f, 1.0f);	// ���ʸ� - ������ �Ķ�
    baseFaceColors[3] = float3(1.0f, 0.8f, 0.2f);	// �����ʸ� - ������ ���
    baseFaceColors[4] = float3(1.0f, 0.2f, 0.8f);	// ���� - ������ ����Ÿ
    baseFaceColors[5] = float3(0.2f, 0.9f, 0.9f);	// �Ʒ��� - ������ �þ�
    
    // �� �麰 ���� �Ӽ�
    float metallicValues[6] = { 0.95f, 0.85f, 0.98f, 0.2f, 0.1f, 0.6f };
    float roughnessValues[6] = { 0.05f, 0.1f, 0.02f, 0.7f, 0.8f, 0.3f };
    
    float3 baseColor = baseFaceColors[faceIndex];
    float metallic = metallicValues[faceIndex];
    float roughness = roughnessValues[faceIndex];
    
    // ���� ���
    float3 faceNormals[6] = {
        float3(0.0, 0.0, -1.0), // �ո�
        float3(0.0, 0.0, 1.0),  // �޸�
        float3(-1.0, 0.0, 0.0), // ���ʸ�
        float3(1.0, 0.0, 0.0),  // �����ʸ�
        float3(0.0, 1.0, 0.0),  // ����
        float3(0.0, -1.0, 0.0)  // �Ʒ���
    };
    
    float3 normal = faceNormals[faceIndex];
    float3 worldPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 viewDir = -normalize(WorldRayDirection());
    
    // ���� ���� �ý���
    float3 finalColor = float3(0, 0, 0);
    
    // ���� ���� (������ ���µ�)
    float3 mainLightPos = float3(6.0f * cos(g_time * 0.3f), 10.0f, 6.0f * sin(g_time * 0.3f));
    float3 mainLightDir = normalize(mainLightPos - worldPos);
    float3 mainLightColor = float3(1.0f, 0.95f, 0.8f) * 3.0f; // ������ �����
    
    // ���� ���� (������ ���µ�)
    float3 fillLightPos = float3(-4.0f, 6.0f, -4.0f);
    float3 fillLightDir = normalize(fillLightPos - worldPos);
    float3 fillLightColor = float3(0.6f, 0.8f, 1.2f) * 1.5f; // ������ ��� ����Ʈ
    
    // ���� ������ ���
    float mainNdotL = max(0.0f, dot(normal, mainLightDir));
    float mainDistance = length(mainLightPos - worldPos);
    float mainAttenuation = 1.0f / (mainDistance * mainDistance * 0.05f + 1.0f);
    
    // ���� ������ ���
    float fillNdotL = max(0.0f, dot(normal, fillLightDir));
    float fillDistance = length(fillLightPos - worldPos);
    float fillAttenuation = 1.0f / (fillDistance * fillDistance * 0.1f + 1.0f);
    
    // Ȯ�� �ݻ�
    float3 diffuse = baseColor * (
        mainLightColor * mainNdotL * mainAttenuation +
        fillLightColor * fillNdotL * fillAttenuation * 0.5f
    );
    
    // ȯ�汤 (HDR ��ī�̶���Ʈ)
    float3 ambient = baseColor * float3(0.15f, 0.2f, 0.35f); // �ϴû� ȯ�汤
    
    // ����ŧ�� �ݻ� (���� ���)
    float3 specular = float3(0, 0, 0);
    
    // ���� ����Ʈ ����ŧ��
    float3 mainHalfVector = normalize(mainLightDir + viewDir);
    float mainNdotH = max(0.0f, dot(normal, mainHalfVector));
    float mainSpecPower = lerp(256.0f, 4.0f, roughness);
    specular += mainLightColor * pow(mainNdotH, mainSpecPower) * mainAttenuation * (1.0f - roughness);
    
    // ���� ����Ʈ ����ŧ��
    float3 fillHalfVector = normalize(fillLightDir + viewDir);
    float fillNdotH = max(0.0f, dot(normal, fillHalfVector));
    float fillSpecPower = lerp(128.0f, 8.0f, roughness);
    specular += fillLightColor * pow(fillNdotH, fillSpecPower) * fillAttenuation * 0.3f;
    
    // �ݻ� ��� (��ǰ��)
    float3 reflectedColor = float3(0, 0, 0);
    if (metallic > 0.1f && payload.recursionDepth < 2) {
        float3 reflectDir = Reflect(-viewDir, normal);
        
        RayDesc reflectRay;
        reflectRay.Origin = worldPos + normal * 0.001f;
        reflectRay.Direction = reflectDir;
        reflectRay.TMin = 0.001f;
        reflectRay.TMax = 1000.0f;
        
        RayPayload reflectPayload;
        reflectPayload.color = float4(0, 0, 0, 1);
        reflectPayload.recursionDepth = payload.recursionDepth + 1;
        
        TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 1, 0, reflectRay, reflectPayload);
        
        reflectedColor = reflectPayload.color.rgb * metallic * (1.0f - roughness);
    }
    
    // ������ ȿ��
    float fresnel = pow(1.0f - max(0.0f, dot(viewDir, normal)), 3.0f);
    float3 fresnelColor = lerp(float3(0.04f, 0.04f, 0.04f), baseColor, metallic);
    
    // ���� ���� �ռ� (���� ��� ������)
    float3 dielectric = diffuse + specular * fresnelColor;
    float3 conductor = baseColor * specular + reflectedColor;
    finalColor = lerp(dielectric, conductor, metallic) + ambient;
    
    // ���鿡 �߱� ȿ�� (�� �ε巴�� ������)
    if (faceIndex == 4) {
        float3 emissionColor = HSVtoRGB(float3(0.8f + sin(g_time) * 0.1f, 0.6f, 0.8f));
        finalColor += emissionColor * (0.5f + 0.3f * sin(g_time * 2.0f));
    }
    
    // �����ڸ� �߱� ȿ��
    float edgeGlow = pow(1.0f - abs(dot(viewDir, normal)), 2.0f);
    finalColor += baseColor * edgeGlow * 0.1f;
    
    payload.color = float4(finalColor, 1.0f);
}

[shader("miss")]
void MissShader(inout RayPayload payload) {
    // ��� ���� ��ī�̹ڽ�
    float3 rayDir = WorldRayDirection();
    
    // �׶���Ʈ �ϴ� (�� �Ƹ��ٿ� ����)
    float t = 0.5f * (rayDir.y + 1.0f);
    
    // �ð��� ���� �ϴû� ��ȭ
    float timePhase = g_time * 0.1f;
    float3 skyHorizon = HSVtoRGB(float3(0.6f + sin(timePhase) * 0.1f, 0.8f, 1.0f));      // �þ�-���
    float3 skyZenith = HSVtoRGB(float3(0.66f + cos(timePhase * 1.3f) * 0.05f, 0.9f, 0.4f)); // ���� �Ķ�
    
    float3 skyColor = lerp(skyHorizon, skyZenith, pow(t, 1.5f));
    
    // ���� �¾� ȿ�� (�� �Ƹ����)
    float3 sunDir1 = normalize(float3(cos(g_time * 0.08f), 0.7f, sin(g_time * 0.08f)));
    float3 sunDir2 = normalize(float3(-cos(g_time * 0.12f + 3.14f), 0.6f, -sin(g_time * 0.12f + 3.14f)));
    
    // ù ��° �¾� (������ ��)
    float sunIntensity1 = pow(max(0.0f, dot(rayDir, sunDir1)), 128.0f);
    float3 sunColor1 = HSVtoRGB(float3(0.1f, 0.8f, 8.0f)) * sunIntensity1;
    
    // �� ��° �¾� (������ ��)
    float sunIntensity2 = pow(max(0.0f, dot(rayDir, sunDir2)), 64.0f);
    float3 sunColor2 = HSVtoRGB(float3(0.55f, 0.7f, 4.0f)) * sunIntensity2;
    
    // ���� (�� ��¦�̴� ȿ��)
    float starField = 0.0f;
    for (int i = 0; i < 3; i++) {
        float2 starCoord = rayDir.xz * (10.0f + float(i) * 5.0f) + float(i) * 100.0f;
        float star = pow(max(0.0f, 
            sin(starCoord.x * 20.0f + g_time) * 
            cos(starCoord.y * 15.0f + g_time * 1.3f)), 20.0f);
        starField += star * (1.0f - t) * 0.3f;
    }
    
    float3 stars = float3(1, 1, 1) * starField;
    
    // �׺�� ȿ�� (���� ���� ����)
    float nebula = sin(rayDir.x * 5.0f + g_time * 0.2f) * 
                   cos(rayDir.y * 7.0f + g_time * 0.15f) * 
                   sin(rayDir.z * 6.0f + g_time * 0.25f);
    nebula = max(0.0f, nebula) * 0.3f;
    float3 nebulaColor = HSVtoRGB(float3(0.8f + nebula * 0.2f, 0.6f, nebula * 2.0f));
    
    payload.color = float4(skyColor + sunColor1 + sunColor2 + stars + nebulaColor, 1.0f);
}
)";

class DXRCubeRenderer {
private:
	static const UINT FRAME_COUNT = 2; // 2 -> 3���� ����

	// DirectX 12 �⺻ ��ü��
	ComPtr<ID3D12Device5> m_device;
	ComPtr<ID3D12GraphicsCommandList4> m_commandList;
	ComPtr<ID3D12CommandQueue> m_commandQueue;
	ComPtr<ID3D12CommandAllocator> m_commandAllocator;
	ComPtr<IDXGISwapChain3> m_swapChain;
	ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
	ComPtr<ID3D12Resource> m_renderTargets[FRAME_COUNT];
	ComPtr<ID3D12Fence> m_fence;

	// ����Ʈ���̽� ����
	ComPtr<ID3D12StateObject> m_raytracingPipelineState;
	ComPtr<ID3D12Resource> m_bottomLevelAS;
	ComPtr<ID3D12Resource> m_topLevelAS;
	ComPtr<ID3D12Resource> m_vertexBuffer;
	ComPtr<ID3D12Resource> m_indexBuffer;
	ComPtr<ID3D12Resource> m_outputResource;
	ComPtr<ID3D12Resource> m_constantBuffer; // ��� ���� �߰�
	ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;
	ComPtr<ID3D12RootSignature> m_globalRootSignature;
	ComPtr<ID3D12RootSignature> m_localRootSignature; // �߰�: ���� ��Ʈ �ñ״�ó

	// ���̴� ���̺� ����
	ComPtr<ID3D12Resource> m_raygenShaderTable;
	ComPtr<ID3D12Resource> m_missShaderTable;
	ComPtr<ID3D12Resource> m_hitGroupShaderTable;
	ComPtr<ID3D12StateObjectProperties> m_stateObjectProperties;

	UINT m_frameIndex = 0;
	UINT64 m_fenceValue = 1;
	HANDLE m_fenceEvent;
	HWND m_hwnd;
	SceneConstants* m_mappedConstantData = nullptr;


	// ������ü ���� ������
	std::vector<Vertex> m_vertices = {
		// �ո�
		{{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}},
		{{-1.0f,  1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}},
		{{ 1.0f,  1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}},
		{{ 1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}},

		// �޸�
		{{ 1.0f, -1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}},
		{{ 1.0f,  1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}},
		{{-1.0f,  1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}},
		{{-1.0f, -1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}}
	};

	std::vector<UINT> m_indices = {
		// �ո�
		0, 1, 2, 0, 2, 3,
		// �޸�
		4, 5, 6, 4, 6, 7,
		// ���ʸ�
		7, 6, 1, 7, 1, 0,
		// �����ʸ�
		3, 2, 5, 3, 5, 4,
		// ����
		1, 6, 5, 1, 5, 2,
		// �Ʒ���
		7, 0, 3, 7, 3, 4
	};

public:
	bool Initialize(HWND hwnd, UINT width, UINT height) {
		m_hwnd = hwnd;

		// DirectX 12 �ʱ�ȭ
		if (!InitializeD3D12(width, height)) {
			return false;
		}

		// DXR ���� Ȯ��
		if (!CheckRaytracingSupport()) {
			std::cout << "DXR�� �������� �ʽ��ϴ�." << std::endl;
			return false;
		}

		// ��� ���� ����
		if (!CreateConstantBuffer()) {
			std::cout << "��� ���� ���� ����" << std::endl;
			return false;
		}

		// ��ũ���� �� ���� �߰�
		if (!CreateDescriptorHeap()) {
			std::cout << "��ũ���� �� ���� ����" << std::endl;
			return false;
		}

		// �۷ι� ��Ʈ �ñ״�ó ����
		if (!CreateGlobalRootSignature()) {
			std::cout << "�۷ι� ��Ʈ �ñ״�ó ���� ����" << std::endl;
			return false;
		}

		// ���� ��Ʈ �ñ״�ó ����
		if (!CreateLocalRootSignature()) {
			std::cout << "���� ��Ʈ �ñ״�ó ���� ����" << std::endl;
			return false;
		}

		// ������ü ������Ʈ�� ����
		if (!CreateCubeGeometry()) {
			return false;
		}

		// ���� ���� ���� �� ����
		if (!CreateAccelerationStructures()) {
			return false;
		}

		// ����Ʈ���̽� ���������� ����
		if (!CreateRaytracingPipeline()) {
			return false;
		}

		// ���̴� ���̺� ����
		if (!CreateShaderTables()) {
			std::cout << "���̴� ���̺� ���� ����" << std::endl;
			return false;
		}

		// ��� ���ҽ� ���� (CBV/UAV ��ũ���� ���� ����)
		if (!CreateOutputResource(width, height)) {
			return false;
		}

		// ��ũ���� ����
		if (!CreateDescriptors()) {
			std::cout << "��ũ���� ���� ����" << std::endl;
			return false;
		}

		return true;
	}

	void Render() {
		// �ð� ������Ʈ
		static auto startTime = std::chrono::high_resolution_clock::now();
		static auto prevTime = startTime;
		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		std::cout << std::format("FPS: {:.0f}\n"
				,1.0f / std::chrono::duration<float, std::chrono::seconds::period>(currentTime - prevTime).count());
		prevTime = currentTime;

		// ��� ���� ������Ʈ
		if (m_mappedConstantData) {
			m_mappedConstantData->time = time;
		}

		// ��� ��� ����
		m_commandAllocator->Reset();
		m_commandList->Reset(m_commandAllocator.Get(), nullptr);

		// ���� ����� �ε���
		m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

		// ���������� ���� ����
		m_commandList->SetPipelineState1(m_raytracingPipelineState.Get());

		// ��Ʈ �ñ״�ó ����
		m_commandList->SetComputeRootSignature(m_globalRootSignature.Get());

		//��ũ���� �� ����
		ID3D12DescriptorHeap* pDescriptorHeaps[] = { m_descriptorHeap.Get() };
		m_commandList->SetDescriptorHeaps(_countof(pDescriptorHeaps), pDescriptorHeaps);
		auto descriptorTableHandle = m_descriptorHeap->GetGPUDescriptorHandleForHeapStart();
		m_commandList->SetComputeRootDescriptorTable(0, descriptorTableHandle);

		// ����Ʈ���̽� ����ġ
		D3D12_DISPATCH_RAYS_DESC dispatchDesc = {
			.RayGenerationShaderRecord = { // Ray Generation ���̴� ���ڵ�
				.StartAddress = static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(m_raygenShaderTable->GetGPUVirtualAddress()),
				.SizeInBytes = kRecord,
			},
			.MissShaderTable = { // Miss ���̴� ���̺�
				.StartAddress = static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(m_missShaderTable->GetGPUVirtualAddress()),
				.SizeInBytes = kRecord,
				.StrideInBytes = kRecord,
			},
			.HitGroupTable = { // Hit Group ���̴� ���̺�
				.StartAddress = static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(m_hitGroupShaderTable->GetGPUVirtualAddress()),
				.SizeInBytes = kRecord,
				.StrideInBytes = kRecord,
			},
			.Width = 2560,
			.Height = 1440,
			.Depth = 1
		};

		// ����Ʈ���̽� ����ġ
		m_commandList->DispatchRays(&dispatchDesc);

		D3D12_RESOURCE_BARRIER uavBarrier = {};
		uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		uavBarrier.UAV.pResource = m_outputResource.Get();
		m_commandList->ResourceBarrier(1, &uavBarrier);

		// ���ҽ� ���� ��ȯ
		D3D12_RESOURCE_BARRIER barriers[2];

		// ��� �ؽ�ó�� ���� �ҽ��� ��ȯ
		barriers[0] = {};
		barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barriers[0].Transition.pResource = m_outputResource.Get();
		barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
		barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

		// ����۸� ���� ������� ��ȯ
		barriers[1] = {};
		barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barriers[1].Transition.pResource = m_renderTargets[m_frameIndex].Get();
		barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
		barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
		barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

		m_commandList->ResourceBarrier(2, barriers);

		D3D12_TEXTURE_COPY_LOCATION srcLocation = {};
		srcLocation.pResource = m_outputResource.Get();
		srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		srcLocation.SubresourceIndex = 0;

		D3D12_TEXTURE_COPY_LOCATION dstLocation = {};
		dstLocation.pResource = m_renderTargets[m_frameIndex].Get();
		dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		dstLocation.SubresourceIndex = 0;

		m_commandList->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, nullptr);

		barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
		barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

		barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;

		m_commandList->ResourceBarrier(2, barriers);

		// ��� ����
		m_commandList->Close();
		ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
		m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

		// ������ ����
		m_swapChain->Present(0, 0);
		WaitForPreviousFrame();
	}

private:
	bool InitializeD3D12(UINT width, UINT height) {
		UINT dxgiFactoryFlags = 0;

#ifdef _DEBUG
		// ����� ���̾� Ȱ��ȭ
		ComPtr<ID3D12Debug> debugController;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
			debugController->EnableDebugLayer();

			// �߰� ������� ���� GPU ��� ��ȿ�� �˻� Ȱ��ȭ
			ComPtr<ID3D12Debug1> debugController1;
			if (SUCCEEDED(debugController->QueryInterface(IID_PPV_ARGS(&debugController1)))) {
				debugController1->SetEnableGPUBasedValidation(TRUE);
			}

			dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
		}
#endif // _DEBUG

		// DXGI ���丮 ����
		ComPtr<IDXGIFactory4> factory;
		if (FAILED(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)))) {
			return false;
		}

		// �ϵ���� ����� ã��
		ComPtr<IDXGIAdapter1> hardwareAdapter;
		GetHardwareAdapter(factory.Get(), &hardwareAdapter);

		// D3D12 ����̽� ����
		if (FAILED(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device)))) {
			return false;
		}

		// ��� ť ����
		D3D12_COMMAND_QUEUE_DESC queueDesc = {};
		queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

		if (FAILED(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)))) {
			return false;
		}

		// ����ü�� ����
		DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
		swapChainDesc.BufferCount = 2;
		swapChainDesc.Width = width;
		swapChainDesc.Height = height;
		swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
		swapChainDesc.SampleDesc.Count = 1;

		ComPtr<IDXGISwapChain1> swapChain;
		if (FAILED(factory->CreateSwapChainForHwnd(m_commandQueue.Get(), m_hwnd, &swapChainDesc, nullptr, nullptr, &swapChain))) {
			return false;
		}

		if (FAILED(swapChain.As(&m_swapChain))) {
			return false;
		}

		// ��� �Ҵ��� ����
		if (FAILED(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator)))) {
			return false;
		}

		// ��� ��� ����
		if (FAILED(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_commandList)))) {
			return false;
		}

		// �潺 ����
		if (FAILED(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) {
			return false;
		}

		m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

		// RTV �� ���� (�߰�)
		D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
		rtvHeapDesc.NumDescriptors = 2;
		rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
		rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		if (FAILED(m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)))) {
			return false;
		}

		// RTV ����
		D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
		for (UINT i = 0; i < 2; i++) {
			if (FAILED(m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_renderTargets[i])))) {
				return false;
			}
			m_device->CreateRenderTargetView(m_renderTargets[i].Get(), nullptr, rtvHandle);
			rtvHandle.ptr += m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		}

		return true;
	}

	bool CheckRaytracingSupport() {
		D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
		if (FAILED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5)))) {
			return false;
		}

		return options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0;
	}

	// �۷ι� ��Ʈ �ñ״�ó ����
	bool CreateGlobalRootSignature() {
		std::vector<D3D12_ROOT_PARAMETER1> rootParameters;
		std::vector<D3D12_DESCRIPTOR_RANGE1> descriptorRanges;

		// SRV ��ũ���� (���� ������)
		D3D12_DESCRIPTOR_RANGE1 srvRange{}; // t0: TLAS
		srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		srvRange.NumDescriptors = 1;
		srvRange.BaseShaderRegister = 0;
		srvRange.RegisterSpace = 0;
		srvRange.OffsetInDescriptorsFromTableStart = 0;

		// ��ũ���� ������ ����

		D3D12_DESCRIPTOR_RANGE1 cbvRange = {}; // b0: SceneConstants
		cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
		cbvRange.NumDescriptors = 1;
		cbvRange.BaseShaderRegister = 0;
		cbvRange.RegisterSpace = 0;
		cbvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

		// UAV ����
		D3D12_DESCRIPTOR_RANGE1 uavRange = {}; // u0: RenderTarget UAV
		uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		uavRange.NumDescriptors = 1;
		uavRange.BaseShaderRegister = 0;
		uavRange.RegisterSpace = 0;
		uavRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

		descriptorRanges.push_back(srvRange);
		descriptorRanges.push_back(cbvRange);
		descriptorRanges.push_back(uavRange);

		D3D12_ROOT_PARAMETER1 descriptorTableParameter = {};
		descriptorTableParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		descriptorTableParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		descriptorTableParameter.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(descriptorRanges.size());
		descriptorTableParameter.DescriptorTable.pDescriptorRanges = descriptorRanges.data();

		rootParameters.push_back(descriptorTableParameter); // 0: SRV, CBV, UAV (���̺�)

		// ��Ʈ �ñ״�ó ����
		D3D12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
		rootSignatureDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
		rootSignatureDesc.Desc_1_1.NumParameters = static_cast<UINT>(rootParameters.size());
		rootSignatureDesc.Desc_1_1.pParameters = rootParameters.data();
		rootSignatureDesc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

		// ��Ʈ �ñ״�ó ����ȭ
		ComPtr<ID3DBlob> signature;
		ComPtr<ID3DBlob> error;
		HRESULT hr = D3D12SerializeVersionedRootSignature(&rootSignatureDesc, &signature, &error);
		if (FAILED(hr)) {
			if (error) {
				std::cout << "��Ʈ �ñ״�ó ����ȭ ����: " << (char*)error->GetBufferPointer() << std::endl;
			}
			return false;
		}

		// ��Ʈ �ñ״�ó ����
		hr = m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_globalRootSignature));
		if (FAILED(hr)) {
			std::cout << "�۷ι� ��Ʈ �ñ״�ó ���� ����: 0x" << std::hex << hr << std::endl;
			return false;
		}

		return true;
	}

	// ���� ��Ʈ �ñ״�ó ���� (���� �߰�)
	bool CreateLocalRootSignature() {
		// ������ ���� ��Ʈ �ñ״�ó (����� �� ��Ʈ �ñ״�ó)
		D3D12_VERSIONED_ROOT_SIGNATURE_DESC localRootSignatureDesc = {};
		localRootSignatureDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
		localRootSignatureDesc.Desc_1_1.NumParameters = 0;
		localRootSignatureDesc.Desc_1_1.pParameters = nullptr;
		localRootSignatureDesc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;

		ComPtr<ID3DBlob> signature;
		ComPtr<ID3DBlob> error;
		HRESULT hr = D3D12SerializeVersionedRootSignature(&localRootSignatureDesc, &signature, &error);
		if (FAILED(hr)) {
			if (error) {
				std::cout << "���� ��Ʈ �ñ״�ó ����ȭ ����: " << (char*)error->GetBufferPointer() << std::endl;
			}
			return false;
		}

		hr = m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_localRootSignature));
		if (FAILED(hr)) {
			std::cout << "���� ��Ʈ �ñ״�ó ���� ����: 0x" << std::hex << hr << std::endl;
			return false;
		}

		return true;
	}

	bool CreateDescriptorHeap() {
		D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
		descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		descriptorHeapDesc.NumDescriptors = 3; // SRV(TLAS), CBV, UAV
		descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

		HRESULT hr = m_device->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&m_descriptorHeap));
		if (FAILED(hr)) {
			std::cout << "��ũ���� �� ���� ����: 0x" << std::hex << hr << std::endl;
			return false;
		}

		return true;
	}

	bool CreateCubeGeometry() {
		// 1) �⺻ ����
		UINT vertexBufferSize = static_cast<UINT>(sizeof(Vertex) * m_vertices.size());
		UINT indexBufferSize = static_cast<UINT>(sizeof(UINT) * m_indices.size());

		// ���� desc / heap ����
		D3D12_RESOURCE_DESC resourceDesc = {};
		resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		resourceDesc.Height = 1;
		resourceDesc.DepthOrArraySize = 1;
		resourceDesc.MipLevels = 1;
		resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
		resourceDesc.SampleDesc.Count = 1;
		resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

		D3D12_HEAP_PROPERTIES defaultHeap{}; defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
		D3D12_HEAP_PROPERTIES uploadHeap{};  uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;

		// 2) Default Heap ����
		HRESULT hr;
		{
			resourceDesc.Width = vertexBufferSize;
			hr = m_device->CreateCommittedResource(
				&defaultHeap, D3D12_HEAP_FLAG_NONE, &resourceDesc,
				D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_vertexBuffer));
			if (FAILED(hr)) return false;

			resourceDesc.Width = indexBufferSize;
			hr = m_device->CreateCommittedResource(
				&defaultHeap, D3D12_HEAP_FLAG_NONE, &resourceDesc,
				D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_indexBuffer));
			if (FAILED(hr)) return false;
		}

		// 3) Upload Heap ���� �� ������ ���ε�
		ComPtr<ID3D12Resource> vertexUploadBuffer, indexUploadBuffer;
		{
			resourceDesc.Width = vertexBufferSize;
			hr = m_device->CreateCommittedResource(
				&uploadHeap, D3D12_HEAP_FLAG_NONE, &resourceDesc,
				D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&vertexUploadBuffer));
			if (FAILED(hr)) return false;

			resourceDesc.Width = indexBufferSize;
			hr = m_device->CreateCommittedResource(
				&uploadHeap, D3D12_HEAP_FLAG_NONE, &resourceDesc,
				D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&indexUploadBuffer));
			if (FAILED(hr)) return false;

			{
				UINT8* p = nullptr; D3D12_RANGE readRange = {};
				vertexUploadBuffer->Map(0, &readRange, reinterpret_cast<void**>(&p));
				memcpy(p, m_vertices.data(), vertexBufferSize);
				vertexUploadBuffer->Unmap(0, nullptr);

				indexUploadBuffer->Map(0, &readRange, reinterpret_cast<void**>(&p));
				memcpy(p, m_indices.data(), indexBufferSize);
				indexUploadBuffer->Unmap(0, nullptr);
			}
		}

		// 4) Copy ��ɾ�� Default Heap�� ������ ����
		{
			m_commandList->CopyBufferRegion(
				m_vertexBuffer.Get(), 0, vertexUploadBuffer.Get(), 0, vertexBufferSize);
			m_commandList->CopyBufferRegion(
				m_indexBuffer.Get(), 0, indexUploadBuffer.Get(), 0, indexBufferSize);

			// �踮��: COPY_DEST -> �б� ����
			D3D12_RESOURCE_BARRIER barriers[2] {};
			barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			barriers[0].Transition.pResource = m_vertexBuffer.Get();
			barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
			barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
			barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;

			barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			barriers[1].Transition.pResource = m_indexBuffer.Get();
			barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
			barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
			barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_INDEX_BUFFER;

			m_commandList->ResourceBarrier(2, barriers);

			m_commandList->Close();
			ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
			m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
			WaitForPreviousFrame();
			m_commandAllocator->Reset();
			m_commandList->Reset(m_commandAllocator.Get(), nullptr);
		}

		return true;
	}

	bool CreateConstantBuffer() {
		// ��� ���� ũ�� (256����Ʈ ����)
		UINT constantBufferSize = (sizeof(SceneConstants) + 255) & ~255;

		D3D12_HEAP_PROPERTIES heapProps = {};
		heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

		D3D12_RESOURCE_DESC resourceDesc = {};
		resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		resourceDesc.Width = constantBufferSize;
		resourceDesc.Height = 1;
		resourceDesc.DepthOrArraySize = 1;
		resourceDesc.MipLevels = 1;
		resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
		resourceDesc.SampleDesc.Count = 1;
		resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

		HRESULT hr = m_device->CreateCommittedResource(
			&heapProps,
			D3D12_HEAP_FLAG_NONE,
			&resourceDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_constantBuffer));

		if (FAILED(hr)) {
			std::cout << "��� ���� ���� ����: 0x" << std::hex << hr << std::endl;
			return false;
		}

		D3D12_RANGE readRange = {};
		hr = m_constantBuffer->Map(0, &readRange, reinterpret_cast<void**>(&m_mappedConstantData));
		if (FAILED(hr)) {
			std::cout << "��� ���� ���� ����" << std::endl;
			return false;
		}

		// �ʱ� ������ ����
		m_mappedConstantData->time = 0.0f;
		m_mappedConstantData->padding[0]
			= m_mappedConstantData->padding[1]
			= m_mappedConstantData->padding[2]
			= 0.0f;

		return true;
	}

	// ������ ���� ���� ���� (������)
	bool CreateAccelerationStructures() {
		try {
			std::cout << "���� ���� ���� ��..." << std::endl;

			// Bottom Level Acceleration Structure (BLAS) ����
			D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc = {};
			geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
			geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
			geometryDesc.Triangles.VertexBuffer.StartAddress = m_vertexBuffer->GetGPUVirtualAddress();
			geometryDesc.Triangles.VertexBuffer.StrideInBytes = sizeof(Vertex);
			geometryDesc.Triangles.VertexCount = static_cast<UINT>(m_vertices.size());
			geometryDesc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
			geometryDesc.Triangles.IndexBuffer = m_indexBuffer->GetGPUVirtualAddress();
			geometryDesc.Triangles.IndexCount = static_cast<UINT>(m_indices.size());
			geometryDesc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;

			D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bottomLevelInputs = {};
			bottomLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
			bottomLevelInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
			bottomLevelInputs.NumDescs = 1;
			bottomLevelInputs.pGeometryDescs = &geometryDesc;

			D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo = {};
			m_device->GetRaytracingAccelerationStructurePrebuildInfo(&bottomLevelInputs, &bottomLevelPrebuildInfo);

			// BLAS ���ҽ� ����
			D3D12_HEAP_PROPERTIES heapProps = {};
			heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

			D3D12_RESOURCE_DESC resourceDesc = {};
			resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			resourceDesc.Width = bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes;
			resourceDesc.Height = 1;
			resourceDesc.DepthOrArraySize = 1;
			resourceDesc.MipLevels = 1;
			resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
			resourceDesc.SampleDesc.Count = 1;
			resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

			HRESULT hr = m_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
				D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&m_bottomLevelAS));
			if (FAILED(hr)) {
				std::cout << "BLAS ���ҽ� ���� ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// BLAS ����� ��ũ��ġ ���� ����
			ComPtr<ID3D12Resource> scratchResource;
			resourceDesc.Width = bottomLevelPrebuildInfo.ScratchDataSizeInBytes;
			hr = m_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
				D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&scratchResource));
			if (FAILED(hr)) {
				std::cout << "BLAS ��ũ��ġ ���� ���� ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// BLAS ���� ����
			D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasBuildDesc = {};
			blasBuildDesc.Inputs = bottomLevelInputs;
			blasBuildDesc.ScratchAccelerationStructureData = scratchResource->GetGPUVirtualAddress();
			blasBuildDesc.DestAccelerationStructureData = m_bottomLevelAS->GetGPUVirtualAddress();

			// BLAS ���� ����
			m_commandList->BuildRaytracingAccelerationStructure(&blasBuildDesc, 0, nullptr);

			// UAV �踮��
			D3D12_RESOURCE_BARRIER uavBarrier = {};
			uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
			uavBarrier.UAV.pResource = m_bottomLevelAS.Get();
			m_commandList->ResourceBarrier(1, &uavBarrier);

			// Top Level Acceleration Structure (TLAS) ����
			D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = {};

			// ȸ�� ��ȯ ��Ʈ���� ���� (X�� 30��, Y�� 45�� ȸ��)
			float angleX = XM_PI / 6.0f; // 30��
			float angleY = XM_PI / 4.0f; // 45��

			// ȸ�� ��Ʈ���� ���
			float cosX = cos(angleX), sinX = sin(angleX);
			float cosY = cos(angleY), sinY = sin(angleY);

			// Y�� ȸ�� �� X�� ȸ���ϴ� �ռ� ��ȯ ��Ʈ����
			instanceDesc.Transform[0][0] = cosY;
			instanceDesc.Transform[0][1] = sinY * sinX;
			instanceDesc.Transform[0][2] = sinY * cosX;

			instanceDesc.Transform[1][0] = 0.0f;
			instanceDesc.Transform[1][1] = cosX;
			instanceDesc.Transform[1][2] = -sinX;

			instanceDesc.Transform[2][0] = -sinY;
			instanceDesc.Transform[2][1] = cosY * sinX;
			instanceDesc.Transform[2][2] = cosY * cosX;

			instanceDesc.InstanceMask = 0xFF;
			instanceDesc.AccelerationStructure = m_bottomLevelAS->GetGPUVirtualAddress();

			// �ν��Ͻ� ���� ���� (������)
			ComPtr<ID3D12Resource> instanceBuffer;
			heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

			// ���ο� ���ҽ� ��ũ���� ����
			D3D12_RESOURCE_DESC instanceResourceDesc = {};
			instanceResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			instanceResourceDesc.Width = sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
			instanceResourceDesc.Height = 1;
			instanceResourceDesc.DepthOrArraySize = 1;
			instanceResourceDesc.MipLevels = 1;
			instanceResourceDesc.Format = DXGI_FORMAT_UNKNOWN;
			instanceResourceDesc.SampleDesc.Count = 1;
			instanceResourceDesc.SampleDesc.Quality = 0;
			instanceResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			instanceResourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

			hr = m_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &instanceResourceDesc,
				D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&instanceBuffer));
			if (FAILED(hr)) {
				std::cout << "�ν��Ͻ� ���� ���� ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// �ν��Ͻ� ������ ����
			void* pInstanceData;
			D3D12_RANGE readRange = {};
			instanceBuffer->Map(0, &readRange, &pInstanceData);
			memcpy(pInstanceData, &instanceDesc, sizeof(instanceDesc));
			instanceBuffer->Unmap(0, nullptr);

			// TLAS �Է� ����
			D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS topLevelInputs = {};
			topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
			topLevelInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
			topLevelInputs.NumDescs = 1;
			topLevelInputs.InstanceDescs = instanceBuffer->GetGPUVirtualAddress();

			D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo = {};
			m_device->GetRaytracingAccelerationStructurePrebuildInfo(&topLevelInputs, &topLevelPrebuildInfo);

			// TLAS ���ҽ� ����
			heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

			D3D12_RESOURCE_DESC tlasResourceDesc = {};
			tlasResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			tlasResourceDesc.Width = topLevelPrebuildInfo.ResultDataMaxSizeInBytes;
			tlasResourceDesc.Height = 1;
			tlasResourceDesc.DepthOrArraySize = 1;
			tlasResourceDesc.MipLevels = 1;
			tlasResourceDesc.Format = DXGI_FORMAT_UNKNOWN;
			tlasResourceDesc.SampleDesc.Count = 1;
			tlasResourceDesc.SampleDesc.Quality = 0;
			tlasResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			tlasResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

			hr = m_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &tlasResourceDesc,
				D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&m_topLevelAS));
			if (FAILED(hr)) {
				std::cout << "TLAS ���ҽ� ���� ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// TLAS ��ũ��ġ ���� ����
			ComPtr<ID3D12Resource> tlasScratchResource;

			D3D12_RESOURCE_DESC tlasScratchResourceDesc = {};
			tlasScratchResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			tlasScratchResourceDesc.Width = topLevelPrebuildInfo.ScratchDataSizeInBytes;
			tlasScratchResourceDesc.Height = 1;
			tlasScratchResourceDesc.DepthOrArraySize = 1;
			tlasScratchResourceDesc.MipLevels = 1;
			tlasScratchResourceDesc.Format = DXGI_FORMAT_UNKNOWN;
			tlasScratchResourceDesc.SampleDesc.Count = 1;
			tlasScratchResourceDesc.SampleDesc.Quality = 0;
			tlasScratchResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			tlasScratchResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

			hr = m_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &tlasScratchResourceDesc,
				D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tlasScratchResource));
			if (FAILED(hr)) {
				std::cout << "TLAS ��ũ��ġ ���� ���� ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// TLAS ����
			D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasBuildDesc = {};
			tlasBuildDesc.Inputs = topLevelInputs;
			tlasBuildDesc.ScratchAccelerationStructureData = tlasScratchResource->GetGPUVirtualAddress();
			tlasBuildDesc.DestAccelerationStructureData = m_topLevelAS->GetGPUVirtualAddress();

			m_commandList->BuildRaytracingAccelerationStructure(&tlasBuildDesc, 0, nullptr);

			// UAV �踮��
			uavBarrier.UAV.pResource = m_topLevelAS.Get();
			m_commandList->ResourceBarrier(1, &uavBarrier);

			// ��� ���� (���� ���� ���带 ����)
			m_commandList->Close();
			ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
			m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

			// GPU �۾� �Ϸ� ���
			WaitForPreviousFrame();

			// ��� ��� ����
			m_commandAllocator->Reset();
			m_commandList->Reset(m_commandAllocator.Get(), nullptr);

			std::cout << "���� ���� ���� ����!" << std::endl;
			return true;
		}
		catch (...) {
			std::cout << "���� ���� ���� �� ���� �߻�" << std::endl;
			return false;
		}
	}

	// DXC�� ����� ����Ʈ���̽� ���������� ���� (���� ������)
	bool CreateRaytracingPipeline() {
		try {
			std::cout << "����Ʈ���̽� ���������� ���� ��..." << std::endl;

			// DXC �����Ϸ� �� ��ƿ��Ƽ ����
			ComPtr<IDxcUtils> dxcUtils;
			ComPtr<IDxcCompiler3> dxcCompiler;

			HRESULT hr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxcUtils));
			if (FAILED(hr)) {
				std::cout << "DxcUtils ���� ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			hr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler));
			if (FAILED(hr)) {
				std::cout << "DxcCompiler ���� ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// �ҽ� �ڵ带 DXC ���۷� ����
			ComPtr<IDxcBlobEncoding> sourceBlob;
			hr = dxcUtils->CreateBlob(g_raytracingShaderCode, static_cast<UINT32>(strlen(g_raytracingShaderCode)), CP_UTF8, &sourceBlob);
			if (FAILED(hr)) {
				std::cout << "�ҽ� ��� ���� ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// ������ �μ� ����
			std::vector<LPCWSTR> arguments;
			arguments.push_back(L"-T");         // Ÿ�� ��������
			arguments.push_back(L"lib_6_3");    // ����Ʈ���̽� ���̺귯�� 6.3
#ifdef _DEBUG
			arguments.push_back(L"-Zi");        // ����� ���� ����
			arguments.push_back(L"-Od");        // ����ȭ ��Ȱ��ȭ
#else
			arguments.push_back(L"-O3");        // ����ȭ ���� 3
#endif // _DEBUG

			DxcBuffer sourceBuffer = {};
			sourceBuffer.Ptr = sourceBlob->GetBufferPointer();
			sourceBuffer.Size = sourceBlob->GetBufferSize();
			sourceBuffer.Encoding = DXC_CP_ACP;

			// ���̴� ������
			ComPtr<IDxcResult> compileResult;
			hr = dxcCompiler->Compile(&sourceBuffer, arguments.data(), static_cast<UINT32>(arguments.size()), nullptr, IID_PPV_ARGS(&compileResult));
			if (FAILED(hr)) {
				std::cout << "���̴� ������ ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// ������ ��� Ȯ��
			HRESULT compileHR;
			compileResult->GetStatus(&compileHR);
			if (FAILED(compileHR)) {
				ComPtr<IDxcBlobEncoding> errorBlob;
				compileResult->GetErrorBuffer(&errorBlob);
				if (errorBlob) {
					std::cout << "���̴� ������ ����: " << (char*)errorBlob->GetBufferPointer() << std::endl;
				}
				return false;
			}

			// �����ϵ� ���̴� ��������
			ComPtr<IDxcBlob> shaderBlob;
			compileResult->GetResult(&shaderBlob);
			if (!shaderBlob) {
				std::cout << "�����ϵ� ���̴��� ������ �� �����ϴ�." << std::endl;
				return false;
			}

			std::cout << "���̴� ������ ����!" << std::endl;

			// ����Ʈ���̽� ���������� ���� ��ü ����
			std::vector<D3D12_STATE_SUBOBJECT> subobjects;

			// 1. DXIL ���̺귯��
			D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {};
			dxilLibDesc.DXILLibrary.pShaderBytecode = shaderBlob->GetBufferPointer();
			dxilLibDesc.DXILLibrary.BytecodeLength = shaderBlob->GetBufferSize();
			dxilLibDesc.NumExports = 0; // ��� �ɺ� ��������
			dxilLibDesc.pExports = nullptr;

			D3D12_STATE_SUBOBJECT dxilLibSubobject = {};
			dxilLibSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
			dxilLibSubobject.pDesc = &dxilLibDesc;
			subobjects.push_back(dxilLibSubobject);

			// 2. ��Ʈ �׷� (Hit Group) ����
			static const WCHAR* hitGroupName = L"HitGroup";
			static const WCHAR* closestHitShaderName = L"ClosestHitShader";

			D3D12_HIT_GROUP_DESC hitGroupDesc = {};
			hitGroupDesc.HitGroupExport = hitGroupName;
			hitGroupDesc.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
			hitGroupDesc.ClosestHitShaderImport = closestHitShaderName;
			hitGroupDesc.AnyHitShaderImport = nullptr;
			hitGroupDesc.IntersectionShaderImport = nullptr;

			D3D12_STATE_SUBOBJECT hitGroupSubobject = {};
			hitGroupSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
			hitGroupSubobject.pDesc = &hitGroupDesc;
			subobjects.push_back(hitGroupSubobject);

			// 3. ���̴� ����
			D3D12_RAYTRACING_SHADER_CONFIG shaderConfig = {};
			shaderConfig.MaxPayloadSizeInBytes = 8 * sizeof(float); // float4 ũ��
			shaderConfig.MaxAttributeSizeInBytes = 2 * sizeof(float); // float2 ũ�� (barycentrics)

			D3D12_STATE_SUBOBJECT shaderConfigSubobject = {};
			shaderConfigSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
			shaderConfigSubobject.pDesc = &shaderConfig;
			subobjects.push_back(shaderConfigSubobject);

			//// 4. ���갴ü�� �ͽ���Ʈ ���� (���� �߰�)
			//static const WCHAR* raygenShaderName = L"RaygenShader";
			//static const WCHAR* missShaderName = L"MissShader";

			//static const WCHAR* shaderExports[] = { raygenShaderName, missShaderName, hitGroupName };

			//D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION shaderConfigAssociation = {};
			//shaderConfigAssociation.pSubobjectToAssociate = &shaderConfigSubobject;
			//shaderConfigAssociation.NumExports = 3; // ��Ȯ�� ���� ����
			//shaderConfigAssociation.pExports = shaderExports;

			//D3D12_STATE_SUBOBJECT shaderConfigAssociationSubobject = {};
			//shaderConfigAssociationSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
			//shaderConfigAssociationSubobject.pDesc = &shaderConfigAssociation;
			//subobjects.push_back(shaderConfigAssociationSubobject);

			//// 5. ���� ��Ʈ �ñ״�ó (���� �߰�)
			//D3D12_LOCAL_ROOT_SIGNATURE localRootSignature = {};
			//localRootSignature.pLocalRootSignature = m_localRootSignature.Get();

			//D3D12_STATE_SUBOBJECT localRootSignatureSubobject = {};
			//localRootSignatureSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
			//localRootSignatureSubobject.pDesc = &localRootSignature;
			//subobjects.push_back(localRootSignatureSubobject);

			//// 6. ���� ��Ʈ �ñ״�ó ����
			//D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION localRootSignatureAssociation = {};
			//localRootSignatureAssociation.pSubobjectToAssociate = &localRootSignatureSubobject;
			//localRootSignatureAssociation.NumExports = 3; // ��Ȯ�� ���� ����
			//localRootSignatureAssociation.pExports = shaderExports;

			//D3D12_STATE_SUBOBJECT localRootSignatureAssociationSubobject = {};
			//localRootSignatureAssociationSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
			//localRootSignatureAssociationSubobject.pDesc = &localRootSignatureAssociation;
			//subobjects.push_back(localRootSignatureAssociationSubobject);

			// 7. �۷ι� ��Ʈ �ñ״�ó ����
			D3D12_GLOBAL_ROOT_SIGNATURE globalRootSignature = {};
			globalRootSignature.pGlobalRootSignature = m_globalRootSignature.Get();

			D3D12_STATE_SUBOBJECT globalRootSignatureSubobject = {};
			globalRootSignatureSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
			globalRootSignatureSubobject.pDesc = &globalRootSignature;
			subobjects.push_back(globalRootSignatureSubobject);

			// 8. ���������� ����
			D3D12_RAYTRACING_PIPELINE_CONFIG pipelineConfig = {};
			pipelineConfig.MaxTraceRecursionDepth = 3; // ��� ����

			D3D12_STATE_SUBOBJECT pipelineConfigSubobject = {};
			pipelineConfigSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
			pipelineConfigSubobject.pDesc = &pipelineConfig;
			subobjects.push_back(pipelineConfigSubobject);

			// ���������� ���� ��ü ����
			D3D12_STATE_OBJECT_DESC stateObjectDesc = {};
			stateObjectDesc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
			stateObjectDesc.NumSubobjects = static_cast<UINT>(subobjects.size());
			stateObjectDesc.pSubobjects = subobjects.data();

			hr = m_device->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&m_raytracingPipelineState));
			if (FAILED(hr)) {
				std::cout << "����Ʈ���̽� ���������� ���� ��ü ���� ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// ���� ��ü �Ӽ� ��������
			hr = m_raytracingPipelineState.As(&m_stateObjectProperties);
			if (FAILED(hr)) {
				std::cout << "���� ��ü �Ӽ� �������� ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			std::cout << "����Ʈ���̽� ���������� ���� ����!" << std::endl;
			return true;

		}
		catch (const std::exception& e) {
			std::cout << "���� �߻�: " << e.what() << std::endl;
			return false;
		}
		catch (...) {
			std::cout << "�� �� ���� ���� �߻�" << std::endl;
			return false;
		}
	}

	// ���̴� ���̺� ����
	bool CreateShaderTables() {
		try {
			std::cout << "���̴� ���̺� ���� ��..." << std::endl;

			// �� �Ӽ�
			D3D12_HEAP_PROPERTIES uploadHeapProperties = {};
			uploadHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
			uploadHeapProperties.CreationNodeMask = 1;
			uploadHeapProperties.VisibleNodeMask = 1;

			// Ray Generation ���̴� ���̺� ����
			D3D12_RESOURCE_DESC raygenShaderTableDesc = {
				.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
				.Alignment = 0,
				.Width = kTable, // Ray Generation ���̴� ���̺� ũ��
				.Height = 1,
				.DepthOrArraySize = 1,
				.MipLevels = 1,
				.Format = DXGI_FORMAT_UNKNOWN,
				.SampleDesc = {.Count = 1, .Quality = 0 }, // ���ø� ����
				.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
				.Flags = D3D12_RESOURCE_FLAG_NONE
			};

			HRESULT hr = m_device->CreateCommittedResource(
				&uploadHeapProperties,
				D3D12_HEAP_FLAG_NONE,
				&raygenShaderTableDesc,
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&m_raygenShaderTable));

			if (FAILED(hr)) {
				std::cout << "Ray Generation ���̴� ���̺� ���� ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// Miss ���̴� ���̺� ����
			hr = m_device->CreateCommittedResource(
				&uploadHeapProperties,
				D3D12_HEAP_FLAG_NONE,
				&raygenShaderTableDesc, // ���� ũ�� ���
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&m_missShaderTable));

			if (FAILED(hr)) {
				std::cout << "Miss ���̴� ���̺� ���� ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// Hit Group ���̴� ���̺� ����
			hr = m_device->CreateCommittedResource(
				&uploadHeapProperties,
				D3D12_HEAP_FLAG_NONE,
				&raygenShaderTableDesc, // ���� ũ�� ���
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&m_hitGroupShaderTable));

			if (FAILED(hr)) {
				std::cout << "Hit Group ���̴� ���̺� ���� ����: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// ���̴� �ĺ��ڸ� ���̺� ����
			void* raygenShaderIdentifier = m_stateObjectProperties->GetShaderIdentifier(L"RaygenShader");
			void* missShaderIdentifier = m_stateObjectProperties->GetShaderIdentifier(L"MissShader");
			void* hitGroupIdentifier = m_stateObjectProperties->GetShaderIdentifier(L"HitGroup");

			if (!raygenShaderIdentifier || !missShaderIdentifier || !hitGroupIdentifier) {
				std::cout << "���̴� �ĺ��� �������� ����" << std::endl;
				return false;
			}

			auto writeRecord = [&](ID3D12Resource* resource, const void* shaderId) -> bool {
				UINT8* pData;
				D3D12_RANGE readRange = {};
				HRESULT hr = resource->Map(0, &readRange, reinterpret_cast<void**>(&pData));
				if (SUCCEEDED(hr)) {
					memcpy(pData, shaderId, kId); // kId�� ���̴� �ĺ����� ũ��
					memset(pData + kId, 0, kTable - kId); // ������ ������ 0���� �ʱ�ȭ
					resource->Unmap(0, nullptr);
					return true;
				}
				else
				{
					std::cout << "���̴� ���̺� ���� ����: 0x" << std::hex << hr << std::endl;
					return false;
				}
				};

			// Ray Generation ���̴� �ĺ��� ����
			if (!writeRecord(m_raygenShaderTable.Get(), raygenShaderIdentifier)) { return false; }

			// Miss ���̴� �ĺ��� ����
			if (!writeRecord(m_missShaderTable.Get(), missShaderIdentifier)) { return false; }

			// Hit Group �ĺ��� ����
			if (!writeRecord(m_hitGroupShaderTable.Get(), hitGroupIdentifier)) { return false; }

			std::cout << "���̴� ���̺� ���� ����!" << std::endl;
			return true;

		}
		catch (...) {
			std::cout << "���̴� ���̺� ���� �� ���� �߻�" << std::endl;
			return false;
		}
	}

	bool CreateOutputResource(UINT width, UINT height) {
		D3D12_RESOURCE_DESC resourceDesc = {};
		resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		resourceDesc.Width = width;
		resourceDesc.Height = height;
		resourceDesc.DepthOrArraySize = 1;
		resourceDesc.MipLevels = 1;
		resourceDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		resourceDesc.SampleDesc.Count = 1;
		resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

		D3D12_HEAP_PROPERTIES heapProps = {};
		heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

		HRESULT hr = m_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_outputResource));
		if (FAILED(hr)) {
			std::cout << "��� ���ҽ� ���� ����: 0x" << std::hex << hr << std::endl;
			return false;
		}

		return true;
	}

	bool CreateDescriptors() {
		auto cpuHandle = m_descriptorHeap->GetCPUDescriptorHandleForHeapStart();
		UINT inc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// 0: TLAS SRV ����(t0)
		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {
			.Format = DXGI_FORMAT_UNKNOWN,
			.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
			.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
			.RaytracingAccelerationStructure = {.Location = m_topLevelAS->GetGPUVirtualAddress() }
		};
		m_device->CreateShaderResourceView(nullptr, &srvDesc, cpuHandle);
		cpuHandle.ptr += inc;

		// 1: CBV ����(b0)
		D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {
			.BufferLocation = m_constantBuffer->GetGPUVirtualAddress(),
			.SizeInBytes = (sizeof(SceneConstants) + 255) & ~255 // 256����Ʈ ����
		};
		m_device->CreateConstantBufferView(&cbvDesc, cpuHandle);
		cpuHandle.ptr += inc;

		// 2: UAV ����(u0)
		D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {
			.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
			.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D
		};
		m_device->CreateUnorderedAccessView(m_outputResource.Get(), nullptr, &uavDesc, cpuHandle);
		return true;
	}

	void GetHardwareAdapter(IDXGIFactory1* pFactory, IDXGIAdapter1** ppAdapter) {
		*ppAdapter = nullptr;

		ComPtr<IDXGIAdapter1> adapter;
		ComPtr<IDXGIFactory6> factory6;

		if (SUCCEEDED(pFactory->QueryInterface(IID_PPV_ARGS(&factory6)))) {
			for (UINT adapterIndex = 0;
				SUCCEEDED(factory6->EnumAdapterByGpuPreference(adapterIndex, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)));
				++adapterIndex) {
				DXGI_ADAPTER_DESC1 desc;
				adapter->GetDesc1(&desc);

				if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
					continue;
				}

				if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) {
					break;
				}
			}
		}

		*ppAdapter = adapter.Detach();
	}

	void WaitForPreviousFrame() {
		const UINT64 fence = m_fenceValue;
		if (FAILED(m_commandQueue->Signal(m_fence.Get(), fence))) {
			return;
		}
		m_fenceValue++;

		if (m_fence->GetCompletedValue() < fence) {
			if (FAILED(m_fence->SetEventOnCompletion(fence, m_fenceEvent))) {
				return;
			}
			WaitForSingleObject(m_fenceEvent, INFINITE);
		}
	}
};

// ������ ���ν���
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	switch (uMsg) {
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	}
	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
	// ������ Ŭ���� ���

	if (AllocConsole())
	{
		FILE* fp;
		freopen_s(&fp, "CONOUT$", "w", stdout);
		freopen_s(&fp, "CONOUT$", "w", stderr);
		freopen_s(&fp, "CONIN$", "r", stdin);
		std::ios::sync_with_stdio();
	}

	const wchar_t CLASS_NAME[] = L"DXR Cube Window";

	WNDCLASS wc = {};
	wc.lpfnWndProc = WindowProc;
	wc.hInstance = hInstance;
	wc.lpszClassName = CLASS_NAME;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

	RegisterClass(&wc);

	// ������ ����
	HWND hwnd = CreateWindowEx(
		0,
		CLASS_NAME,
		L"DXR ������ü ������",
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, 2560, 1440,
		NULL,
		NULL,
		hInstance,
		NULL
	);

	if (hwnd == NULL) {
		std::cout << "������ ���� ����" << std::endl;
		std::cout << "�ƹ� Ű�� ������ ����˴ϴ�..." << std::endl;
		std::cin.get();
		return 0;
	}

	ShowWindow(hwnd, nCmdShow);

	// DXR ������ �ʱ�ȭ
	DXRCubeRenderer renderer;
	if (!renderer.Initialize(hwnd, 2560, 1440)) {
		std::cout << "DXR ������ �ʱ�ȭ ����" << std::endl;
		std::cout << "�ƹ� Ű�� ������ ����˴ϴ�..." << std::endl;
		std::cin.get();
		return -1;
	}

	std::cout << "DXR ������ü �������� ���۵Ǿ����ϴ�." << std::endl;
	std::cout << "�����츦 ������ ���α׷��� ����˴ϴ�." << std::endl;

	// �޽��� ����
	MSG msg = {};
	bool running = true;

	while (running)
	{
		while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
			if (msg.message == WM_QUIT) {
				running = false;
				break;
			}
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		// ������
		renderer.Render();
	}

	std::cout << "���α׷��� ����˴ϴ�." << std::endl;
	std::cout << "�ƹ� Ű�� ������ �ܼ�â�� �����ϴ�..." << std::endl;
	std::cin.get();

	return 0;
}