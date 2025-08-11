#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxcapi.h>  // DXC 헤더 추가
#include <DirectXMath.h>
#include <wrl/client.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <atlbase.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxcompiler.lib")  // DXC 라이브러리

using namespace DirectX;
using Microsoft::WRL::ComPtr;

// 정점 구조체
struct Vertex {
	XMFLOAT3 position;
	XMFLOAT3 normal;
};

struct SceneConstants {
	float time;
	float padding[3];
};

constexpr UINT kId = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;		// 32
constexpr UINT kRecord = ((kId + 31) & ~31);					// 32 정렬
constexpr UINT kTable = ((kRecord + 63) & ~63);					// 64 정렬 권장

// 레이트레이싱 셰이더 코드 (수정됨)
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

// 반사 벡터 계산
float3 Reflect(float3 incident, float3 normal) {
    return incident - 2.0f * dot(incident, normal) * normal;
}

// HSV to RGB 색상 변환 (더 생동감 있는 색상을 위해)
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
    
    // 시간 기반 카메라 애니메이션
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
    
    // 시야각 설정
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
    
    // 초기 페이로드 설정
    RayPayload payload;
    payload.color = float4(0, 0, 0, 1);
    payload.recursionDepth = 0;
    
    TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 1, 0, ray, payload);
    
    // 향상된 톤 매핑 (ACES 톤매핑 근사)
    float3 finalColor = payload.color.rgb;
    
    // 노출 조정
    finalColor *= 1.2f;
    
    // ACES 톤매핑 근사
    float3 a = finalColor * 2.51f;
    float3 b = finalColor * 0.03f + 0.59f;
    float3 c = finalColor * 2.43f + 0.14f;
    finalColor = saturate((a) / (b + c));
    
    // 감마 보정
    finalColor = pow(finalColor, float3(1.0f/2.2f, 1.0f/2.2f, 1.0f/2.2f));
    
    RenderTarget[index.xy] = float4(finalColor, 1.0f);
}

[shader("closesthit")]
void ClosestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    // 최대 재귀 깊이 제한
    if (payload.recursionDepth >= 2) {
        payload.color = float4(0.05, 0.05, 0.1, 1);
        return;
    }
    
    uint primitiveIndex = PrimitiveIndex();
    uint faceIndex = primitiveIndex / 2;
        
    // 각 면별 고급 색상과 재질 (훨씬 더 아름다운 색상들)
    float3 baseFaceColors[6];
    
    baseFaceColors[0] = float3(1.0f, 0.2f, 0.2f);	// 앞면 - 선명한 빨강
    baseFaceColors[1] = float3(0.2f, 1.0f, 0.2f);	// 뒷면 - 선명한 초록
    baseFaceColors[2] = float3(0.2f, 0.4f, 1.0f);	// 왼쪽면 - 선명한 파랑
    baseFaceColors[3] = float3(1.0f, 0.8f, 0.2f);	// 오른쪽면 - 선명한 노랑
    baseFaceColors[4] = float3(1.0f, 0.2f, 0.8f);	// 윗면 - 선명한 마젠타
    baseFaceColors[5] = float3(0.2f, 0.9f, 0.9f);	// 아랫면 - 선명한 시안
    
    // 각 면별 재질 속성
    float metallicValues[6] = { 0.95f, 0.85f, 0.98f, 0.2f, 0.1f, 0.6f };
    float roughnessValues[6] = { 0.05f, 0.1f, 0.02f, 0.7f, 0.8f, 0.3f };
    
    float3 baseColor = baseFaceColors[faceIndex];
    float metallic = metallicValues[faceIndex];
    float roughness = roughnessValues[faceIndex];
    
    // 법선 계산
    float3 faceNormals[6] = {
        float3(0.0, 0.0, -1.0), // 앞면
        float3(0.0, 0.0, 1.0),  // 뒷면
        float3(-1.0, 0.0, 0.0), // 왼쪽면
        float3(1.0, 0.0, 0.0),  // 오른쪽면
        float3(0.0, 1.0, 0.0),  // 윗면
        float3(0.0, -1.0, 0.0)  // 아랫면
    };
    
    float3 normal = faceNormals[faceIndex];
    float3 worldPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 viewDir = -normalize(WorldRayDirection());
    
    // 다중 조명 시스템
    float3 finalColor = float3(0, 0, 0);
    
    // 메인 조명 (따뜻한 색온도)
    float3 mainLightPos = float3(6.0f * cos(g_time * 0.3f), 10.0f, 6.0f * sin(g_time * 0.3f));
    float3 mainLightDir = normalize(mainLightPos - worldPos);
    float3 mainLightColor = float3(1.0f, 0.95f, 0.8f) * 3.0f; // 따뜻한 백색광
    
    // 보조 조명 (차가운 색온도)
    float3 fillLightPos = float3(-4.0f, 6.0f, -4.0f);
    float3 fillLightDir = normalize(fillLightPos - worldPos);
    float3 fillLightColor = float3(0.6f, 0.8f, 1.2f) * 1.5f; // 차가운 블루 라이트
    
    // 메인 라이팅 계산
    float mainNdotL = max(0.0f, dot(normal, mainLightDir));
    float mainDistance = length(mainLightPos - worldPos);
    float mainAttenuation = 1.0f / (mainDistance * mainDistance * 0.05f + 1.0f);
    
    // 보조 라이팅 계산
    float fillNdotL = max(0.0f, dot(normal, fillLightDir));
    float fillDistance = length(fillLightPos - worldPos);
    float fillAttenuation = 1.0f / (fillDistance * fillDistance * 0.1f + 1.0f);
    
    // 확산 반사
    float3 diffuse = baseColor * (
        mainLightColor * mainNdotL * mainAttenuation +
        fillLightColor * fillNdotL * fillAttenuation * 0.5f
    );
    
    // 환경광 (HDR 스카이라이트)
    float3 ambient = baseColor * float3(0.15f, 0.2f, 0.35f); // 하늘색 환경광
    
    // 스펙큘러 반사 (물리 기반)
    float3 specular = float3(0, 0, 0);
    
    // 메인 라이트 스펙큘러
    float3 mainHalfVector = normalize(mainLightDir + viewDir);
    float mainNdotH = max(0.0f, dot(normal, mainHalfVector));
    float mainSpecPower = lerp(256.0f, 4.0f, roughness);
    specular += mainLightColor * pow(mainNdotH, mainSpecPower) * mainAttenuation * (1.0f - roughness);
    
    // 보조 라이트 스펙큘러
    float3 fillHalfVector = normalize(fillLightDir + viewDir);
    float fillNdotH = max(0.0f, dot(normal, fillHalfVector));
    float fillSpecPower = lerp(128.0f, 8.0f, roughness);
    specular += fillLightColor * pow(fillNdotH, fillSpecPower) * fillAttenuation * 0.3f;
    
    // 반사 계산 (고품질)
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
    
    // 프레넬 효과
    float fresnel = pow(1.0f - max(0.0f, dot(viewDir, normal)), 3.0f);
    float3 fresnelColor = lerp(float3(0.04f, 0.04f, 0.04f), baseColor, metallic);
    
    // 최종 색상 합성 (물리 기반 렌더링)
    float3 dielectric = diffuse + specular * fresnelColor;
    float3 conductor = baseColor * specular + reflectedColor;
    finalColor = lerp(dielectric, conductor, metallic) + ambient;
    
    // 윗면에 발광 효과 (더 부드럽고 따뜻한)
    if (faceIndex == 4) {
        float3 emissionColor = HSVtoRGB(float3(0.8f + sin(g_time) * 0.1f, 0.6f, 0.8f));
        finalColor += emissionColor * (0.5f + 0.3f * sin(g_time * 2.0f));
    }
    
    // 가장자리 발광 효과
    float edgeGlow = pow(1.0f - abs(dot(viewDir, normal)), 2.0f);
    finalColor += baseColor * edgeGlow * 0.1f;
    
    payload.color = float4(finalColor, 1.0f);
}

[shader("miss")]
void MissShader(inout RayPayload payload) {
    // 고급 동적 스카이박스
    float3 rayDir = WorldRayDirection();
    
    // 그라디언트 하늘 (더 아름다운 색상)
    float t = 0.5f * (rayDir.y + 1.0f);
    
    // 시간에 따른 하늘색 변화
    float timePhase = g_time * 0.1f;
    float3 skyHorizon = HSVtoRGB(float3(0.6f + sin(timePhase) * 0.1f, 0.8f, 1.0f));      // 시안-블루
    float3 skyZenith = HSVtoRGB(float3(0.66f + cos(timePhase * 1.3f) * 0.05f, 0.9f, 0.4f)); // 깊은 파랑
    
    float3 skyColor = lerp(skyHorizon, skyZenith, pow(t, 1.5f));
    
    // 다중 태양 효과 (더 아름답게)
    float3 sunDir1 = normalize(float3(cos(g_time * 0.08f), 0.7f, sin(g_time * 0.08f)));
    float3 sunDir2 = normalize(float3(-cos(g_time * 0.12f + 3.14f), 0.6f, -sin(g_time * 0.12f + 3.14f)));
    
    // 첫 번째 태양 (따뜻한 색)
    float sunIntensity1 = pow(max(0.0f, dot(rayDir, sunDir1)), 128.0f);
    float3 sunColor1 = HSVtoRGB(float3(0.1f, 0.8f, 8.0f)) * sunIntensity1;
    
    // 두 번째 태양 (차가운 색)
    float sunIntensity2 = pow(max(0.0f, dot(rayDir, sunDir2)), 64.0f);
    float3 sunColor2 = HSVtoRGB(float3(0.55f, 0.7f, 4.0f)) * sunIntensity2;
    
    // 별들 (더 반짝이는 효과)
    float starField = 0.0f;
    for (int i = 0; i < 3; i++) {
        float2 starCoord = rayDir.xz * (10.0f + float(i) * 5.0f) + float(i) * 100.0f;
        float star = pow(max(0.0f, 
            sin(starCoord.x * 20.0f + g_time) * 
            cos(starCoord.y * 15.0f + g_time * 1.3f)), 20.0f);
        starField += star * (1.0f - t) * 0.3f;
    }
    
    float3 stars = float3(1, 1, 1) * starField;
    
    // 네뷸라 효과 (성운 같은 색상)
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
	static const UINT FRAME_COUNT = 2; // 2 -> 3으로 변경

	// DirectX 12 기본 객체들
	ComPtr<ID3D12Device5> m_device;
	ComPtr<ID3D12GraphicsCommandList4> m_commandList;
	ComPtr<ID3D12CommandQueue> m_commandQueue;
	ComPtr<ID3D12CommandAllocator> m_commandAllocator;
	ComPtr<IDXGISwapChain3> m_swapChain;
	ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
	ComPtr<ID3D12Resource> m_renderTargets[FRAME_COUNT];
	ComPtr<ID3D12Fence> m_fence;

	// 레이트레이싱 관련
	ComPtr<ID3D12StateObject> m_raytracingPipelineState;
	ComPtr<ID3D12Resource> m_bottomLevelAS;
	ComPtr<ID3D12Resource> m_topLevelAS;
	ComPtr<ID3D12Resource> m_vertexBuffer;
	ComPtr<ID3D12Resource> m_indexBuffer;
	ComPtr<ID3D12Resource> m_outputResource;
	ComPtr<ID3D12Resource> m_constantBuffer; // 멤버 변수 추가
	ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;
	ComPtr<ID3D12RootSignature> m_globalRootSignature;
	ComPtr<ID3D12RootSignature> m_localRootSignature; // 추가: 로컬 루트 시그니처

	// 셰이더 테이블 관련
	ComPtr<ID3D12Resource> m_raygenShaderTable;
	ComPtr<ID3D12Resource> m_missShaderTable;
	ComPtr<ID3D12Resource> m_hitGroupShaderTable;
	ComPtr<ID3D12StateObjectProperties> m_stateObjectProperties;

	UINT m_frameIndex = 0;
	UINT64 m_fenceValue = 1;
	HANDLE m_fenceEvent;
	HWND m_hwnd;
	SceneConstants* m_mappedConstantData = nullptr;


	// 정육면체 정점 데이터
	std::vector<Vertex> m_vertices = {
		// 앞면
		{{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}},
		{{-1.0f,  1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}},
		{{ 1.0f,  1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}},
		{{ 1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}},

		// 뒷면
		{{ 1.0f, -1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}},
		{{ 1.0f,  1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}},
		{{-1.0f,  1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}},
		{{-1.0f, -1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}}
	};

	std::vector<UINT> m_indices = {
		// 앞면
		0, 1, 2, 0, 2, 3,
		// 뒷면
		4, 5, 6, 4, 6, 7,
		// 왼쪽면
		7, 6, 1, 7, 1, 0,
		// 오른쪽면
		3, 2, 5, 3, 5, 4,
		// 윗면
		1, 6, 5, 1, 5, 2,
		// 아랫면
		7, 0, 3, 7, 3, 4
	};

public:
	bool Initialize(HWND hwnd, UINT width, UINT height) {
		m_hwnd = hwnd;

		// DirectX 12 초기화
		if (!InitializeD3D12(width, height)) {
			return false;
		}

		// DXR 지원 확인
		if (!CheckRaytracingSupport()) {
			std::cout << "DXR이 지원되지 않습니다." << std::endl;
			return false;
		}

		// 상수 버퍼 생성
		if (!CreateConstantBuffer()) {
			std::cout << "상수 버퍼 생성 실패" << std::endl;
			return false;
		}

		// 디스크립터 힙 생성 추가
		if (!CreateDescriptorHeap()) {
			std::cout << "디스크립터 힙 생성 실패" << std::endl;
			return false;
		}

		// 글로벌 루트 시그니처 생성
		if (!CreateGlobalRootSignature()) {
			std::cout << "글로벌 루트 시그니처 생성 실패" << std::endl;
			return false;
		}

		// 로컬 루트 시그니처 생성
		if (!CreateLocalRootSignature()) {
			std::cout << "로컬 루트 시그니처 생성 실패" << std::endl;
			return false;
		}

		// 정육면체 지오메트리 생성
		if (!CreateCubeGeometry()) {
			return false;
		}

		// 가속 구조 생성 및 빌드
		if (!CreateAccelerationStructures()) {
			return false;
		}

		// 레이트레이싱 파이프라인 생성
		if (!CreateRaytracingPipeline()) {
			return false;
		}

		// 셰이더 테이블 생성
		if (!CreateShaderTables()) {
			std::cout << "셰이더 테이블 생성 실패" << std::endl;
			return false;
		}

		// 출력 리소스 생성 (CBV/UAV 디스크립터 생성 포함)
		if (!CreateOutputResource(width, height)) {
			return false;
		}

		// 디스크립터 생성
		if (!CreateDescriptors()) {
			std::cout << "디스크립터 생성 실패" << std::endl;
			return false;
		}

		return true;
	}

	void Render() {
		// 시간 업데이트
		static auto startTime = std::chrono::high_resolution_clock::now();
		static auto prevTime = startTime;
		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		std::cout << std::format("FPS: {:.0f}\n"
				,1.0f / std::chrono::duration<float, std::chrono::seconds::period>(currentTime - prevTime).count());
		prevTime = currentTime;

		// 상수 버퍼 업데이트
		if (m_mappedConstantData) {
			m_mappedConstantData->time = time;
		}

		// 명령 목록 리셋
		m_commandAllocator->Reset();
		m_commandList->Reset(m_commandAllocator.Get(), nullptr);

		// 현재 백버퍼 인덱스
		m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

		// 파이프라인 상태 설정
		m_commandList->SetPipelineState1(m_raytracingPipelineState.Get());

		// 루트 시그니처 설정
		m_commandList->SetComputeRootSignature(m_globalRootSignature.Get());

		//디스크립터 힙 설정
		ID3D12DescriptorHeap* pDescriptorHeaps[] = { m_descriptorHeap.Get() };
		m_commandList->SetDescriptorHeaps(_countof(pDescriptorHeaps), pDescriptorHeaps);
		auto descriptorTableHandle = m_descriptorHeap->GetGPUDescriptorHandleForHeapStart();
		m_commandList->SetComputeRootDescriptorTable(0, descriptorTableHandle);

		// 레이트레이싱 디스패치
		D3D12_DISPATCH_RAYS_DESC dispatchDesc = {
			.RayGenerationShaderRecord = { // Ray Generation 셰이더 레코드
				.StartAddress = static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(m_raygenShaderTable->GetGPUVirtualAddress()),
				.SizeInBytes = kRecord,
			},
			.MissShaderTable = { // Miss 셰이더 테이블
				.StartAddress = static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(m_missShaderTable->GetGPUVirtualAddress()),
				.SizeInBytes = kRecord,
				.StrideInBytes = kRecord,
			},
			.HitGroupTable = { // Hit Group 셰이더 테이블
				.StartAddress = static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(m_hitGroupShaderTable->GetGPUVirtualAddress()),
				.SizeInBytes = kRecord,
				.StrideInBytes = kRecord,
			},
			.Width = 2560,
			.Height = 1440,
			.Depth = 1
		};

		// 레이트레이싱 디스패치
		m_commandList->DispatchRays(&dispatchDesc);

		D3D12_RESOURCE_BARRIER uavBarrier = {};
		uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		uavBarrier.UAV.pResource = m_outputResource.Get();
		m_commandList->ResourceBarrier(1, &uavBarrier);

		// 리소스 상태 전환
		D3D12_RESOURCE_BARRIER barriers[2];

		// 출력 텍스처를 복사 소스로 전환
		barriers[0] = {};
		barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barriers[0].Transition.pResource = m_outputResource.Get();
		barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
		barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

		// 백버퍼를 복사 대상으로 전환
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

		// 명령 실행
		m_commandList->Close();
		ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
		m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

		// 프레임 제출
		m_swapChain->Present(0, 0);
		WaitForPreviousFrame();
	}

private:
	bool InitializeD3D12(UINT width, UINT height) {
		UINT dxgiFactoryFlags = 0;

#ifdef _DEBUG
		// 디버그 레이어 활성화
		ComPtr<ID3D12Debug> debugController;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
			debugController->EnableDebugLayer();

			// 추가 디버깅을 위해 GPU 기반 유효성 검사 활성화
			ComPtr<ID3D12Debug1> debugController1;
			if (SUCCEEDED(debugController->QueryInterface(IID_PPV_ARGS(&debugController1)))) {
				debugController1->SetEnableGPUBasedValidation(TRUE);
			}

			dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
		}
#endif // _DEBUG

		// DXGI 팩토리 생성
		ComPtr<IDXGIFactory4> factory;
		if (FAILED(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)))) {
			return false;
		}

		// 하드웨어 어댑터 찾기
		ComPtr<IDXGIAdapter1> hardwareAdapter;
		GetHardwareAdapter(factory.Get(), &hardwareAdapter);

		// D3D12 디바이스 생성
		if (FAILED(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device)))) {
			return false;
		}

		// 명령 큐 생성
		D3D12_COMMAND_QUEUE_DESC queueDesc = {};
		queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

		if (FAILED(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)))) {
			return false;
		}

		// 스왑체인 생성
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

		// 명령 할당자 생성
		if (FAILED(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator)))) {
			return false;
		}

		// 명령 목록 생성
		if (FAILED(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_commandList)))) {
			return false;
		}

		// 펜스 생성
		if (FAILED(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) {
			return false;
		}

		m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

		// RTV 힙 생성 (추가)
		D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
		rtvHeapDesc.NumDescriptors = 2;
		rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
		rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		if (FAILED(m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)))) {
			return false;
		}

		// RTV 생성
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

	// 글로벌 루트 시그니처 생성
	bool CreateGlobalRootSignature() {
		std::vector<D3D12_ROOT_PARAMETER1> rootParameters;
		std::vector<D3D12_DESCRIPTOR_RANGE1> descriptorRanges;

		// SRV 디스크립터 (가속 구조용)
		D3D12_DESCRIPTOR_RANGE1 srvRange{}; // t0: TLAS
		srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		srvRange.NumDescriptors = 1;
		srvRange.BaseShaderRegister = 0;
		srvRange.RegisterSpace = 0;
		srvRange.OffsetInDescriptorsFromTableStart = 0;

		// 디스크립터 레인지 설정

		D3D12_DESCRIPTOR_RANGE1 cbvRange = {}; // b0: SceneConstants
		cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
		cbvRange.NumDescriptors = 1;
		cbvRange.BaseShaderRegister = 0;
		cbvRange.RegisterSpace = 0;
		cbvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

		// UAV 범위
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

		rootParameters.push_back(descriptorTableParameter); // 0: SRV, CBV, UAV (테이블)

		// 루트 시그니처 설명
		D3D12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
		rootSignatureDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
		rootSignatureDesc.Desc_1_1.NumParameters = static_cast<UINT>(rootParameters.size());
		rootSignatureDesc.Desc_1_1.pParameters = rootParameters.data();
		rootSignatureDesc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

		// 루트 시그니처 직렬화
		ComPtr<ID3DBlob> signature;
		ComPtr<ID3DBlob> error;
		HRESULT hr = D3D12SerializeVersionedRootSignature(&rootSignatureDesc, &signature, &error);
		if (FAILED(hr)) {
			if (error) {
				std::cout << "루트 시그니처 직렬화 오류: " << (char*)error->GetBufferPointer() << std::endl;
			}
			return false;
		}

		// 루트 시그니처 생성
		hr = m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_globalRootSignature));
		if (FAILED(hr)) {
			std::cout << "글로벌 루트 시그니처 생성 실패: 0x" << std::hex << hr << std::endl;
			return false;
		}

		return true;
	}

	// 로컬 루트 시그니처 생성 (새로 추가)
	bool CreateLocalRootSignature() {
		// 간단한 로컬 루트 시그니처 (현재는 빈 루트 시그니처)
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
				std::cout << "로컬 루트 시그니처 직렬화 오류: " << (char*)error->GetBufferPointer() << std::endl;
			}
			return false;
		}

		hr = m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_localRootSignature));
		if (FAILED(hr)) {
			std::cout << "로컬 루트 시그니처 생성 실패: 0x" << std::hex << hr << std::endl;
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
			std::cout << "디스크립터 힙 생성 실패: 0x" << std::hex << hr << std::endl;
			return false;
		}

		return true;
	}

	bool CreateCubeGeometry() {
		// 1) 기본 변수
		UINT vertexBufferSize = static_cast<UINT>(sizeof(Vertex) * m_vertices.size());
		UINT indexBufferSize = static_cast<UINT>(sizeof(UINT) * m_indices.size());

		// 공통 desc / heap 설정
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

		// 2) Default Heap 생성
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

		// 3) Upload Heap 생성 및 데이터 업로드
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

		// 4) Copy 명령어로 Default Heap에 데이터 복사
		{
			m_commandList->CopyBufferRegion(
				m_vertexBuffer.Get(), 0, vertexUploadBuffer.Get(), 0, vertexBufferSize);
			m_commandList->CopyBufferRegion(
				m_indexBuffer.Get(), 0, indexUploadBuffer.Get(), 0, indexBufferSize);

			// 배리어: COPY_DEST -> 읽기 상태
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
		// 상수 버퍼 크기 (256바이트 정렬)
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
			std::cout << "상수 버퍼 생성 실패: 0x" << std::hex << hr << std::endl;
			return false;
		}

		D3D12_RANGE readRange = {};
		hr = m_constantBuffer->Map(0, &readRange, reinterpret_cast<void**>(&m_mappedConstantData));
		if (FAILED(hr)) {
			std::cout << "상수 버퍼 맵핑 실패" << std::endl;
			return false;
		}

		// 초기 데이터 설정
		m_mappedConstantData->time = 0.0f;
		m_mappedConstantData->padding[0]
			= m_mappedConstantData->padding[1]
			= m_mappedConstantData->padding[2]
			= 0.0f;

		return true;
	}

	// 완전한 가속 구조 생성 (수정됨)
	bool CreateAccelerationStructures() {
		try {
			std::cout << "가속 구조 생성 중..." << std::endl;

			// Bottom Level Acceleration Structure (BLAS) 생성
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

			// BLAS 리소스 생성
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
				std::cout << "BLAS 리소스 생성 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// BLAS 빌드용 스크래치 버퍼 생성
			ComPtr<ID3D12Resource> scratchResource;
			resourceDesc.Width = bottomLevelPrebuildInfo.ScratchDataSizeInBytes;
			hr = m_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
				D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&scratchResource));
			if (FAILED(hr)) {
				std::cout << "BLAS 스크래치 버퍼 생성 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// BLAS 빌드 설명
			D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasBuildDesc = {};
			blasBuildDesc.Inputs = bottomLevelInputs;
			blasBuildDesc.ScratchAccelerationStructureData = scratchResource->GetGPUVirtualAddress();
			blasBuildDesc.DestAccelerationStructureData = m_bottomLevelAS->GetGPUVirtualAddress();

			// BLAS 빌드 실행
			m_commandList->BuildRaytracingAccelerationStructure(&blasBuildDesc, 0, nullptr);

			// UAV 배리어
			D3D12_RESOURCE_BARRIER uavBarrier = {};
			uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
			uavBarrier.UAV.pResource = m_bottomLevelAS.Get();
			m_commandList->ResourceBarrier(1, &uavBarrier);

			// Top Level Acceleration Structure (TLAS) 생성
			D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = {};

			// 회전 변환 매트릭스 생성 (X축 30도, Y축 45도 회전)
			float angleX = XM_PI / 6.0f; // 30도
			float angleY = XM_PI / 4.0f; // 45도

			// 회전 매트릭스 계산
			float cosX = cos(angleX), sinX = sin(angleX);
			float cosY = cos(angleY), sinY = sin(angleY);

			// Y축 회전 후 X축 회전하는 합성 변환 매트릭스
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

			// 인스턴스 버퍼 생성 (수정됨)
			ComPtr<ID3D12Resource> instanceBuffer;
			heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

			// 새로운 리소스 디스크립터 생성
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
				std::cout << "인스턴스 버퍼 생성 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// 인스턴스 데이터 복사
			void* pInstanceData;
			D3D12_RANGE readRange = {};
			instanceBuffer->Map(0, &readRange, &pInstanceData);
			memcpy(pInstanceData, &instanceDesc, sizeof(instanceDesc));
			instanceBuffer->Unmap(0, nullptr);

			// TLAS 입력 설정
			D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS topLevelInputs = {};
			topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
			topLevelInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
			topLevelInputs.NumDescs = 1;
			topLevelInputs.InstanceDescs = instanceBuffer->GetGPUVirtualAddress();

			D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo = {};
			m_device->GetRaytracingAccelerationStructurePrebuildInfo(&topLevelInputs, &topLevelPrebuildInfo);

			// TLAS 리소스 생성
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
				std::cout << "TLAS 리소스 생성 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// TLAS 스크래치 버퍼 생성
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
				std::cout << "TLAS 스크래치 버퍼 생성 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// TLAS 빌드
			D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasBuildDesc = {};
			tlasBuildDesc.Inputs = topLevelInputs;
			tlasBuildDesc.ScratchAccelerationStructureData = tlasScratchResource->GetGPUVirtualAddress();
			tlasBuildDesc.DestAccelerationStructureData = m_topLevelAS->GetGPUVirtualAddress();

			m_commandList->BuildRaytracingAccelerationStructure(&tlasBuildDesc, 0, nullptr);

			// UAV 배리어
			uavBarrier.UAV.pResource = m_topLevelAS.Get();
			m_commandList->ResourceBarrier(1, &uavBarrier);

			// 명령 실행 (가속 구조 빌드를 위해)
			m_commandList->Close();
			ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
			m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

			// GPU 작업 완료 대기
			WaitForPreviousFrame();

			// 명령 목록 리셋
			m_commandAllocator->Reset();
			m_commandList->Reset(m_commandAllocator.Get(), nullptr);

			std::cout << "가속 구조 생성 성공!" << std::endl;
			return true;
		}
		catch (...) {
			std::cout << "가속 구조 생성 중 예외 발생" << std::endl;
			return false;
		}
	}

	// DXC를 사용한 레이트레이싱 파이프라인 생성 (완전 수정됨)
	bool CreateRaytracingPipeline() {
		try {
			std::cout << "레이트레이싱 파이프라인 생성 중..." << std::endl;

			// DXC 컴파일러 및 유틸리티 생성
			ComPtr<IDxcUtils> dxcUtils;
			ComPtr<IDxcCompiler3> dxcCompiler;

			HRESULT hr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxcUtils));
			if (FAILED(hr)) {
				std::cout << "DxcUtils 생성 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			hr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler));
			if (FAILED(hr)) {
				std::cout << "DxcCompiler 생성 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// 소스 코드를 DXC 버퍼로 생성
			ComPtr<IDxcBlobEncoding> sourceBlob;
			hr = dxcUtils->CreateBlob(g_raytracingShaderCode, static_cast<UINT32>(strlen(g_raytracingShaderCode)), CP_UTF8, &sourceBlob);
			if (FAILED(hr)) {
				std::cout << "소스 블롭 생성 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// 컴파일 인수 설정
			std::vector<LPCWSTR> arguments;
			arguments.push_back(L"-T");         // 타겟 프로파일
			arguments.push_back(L"lib_6_3");    // 레이트레이싱 라이브러리 6.3
#ifdef _DEBUG
			arguments.push_back(L"-Zi");        // 디버그 정보 포함
			arguments.push_back(L"-Od");        // 최적화 비활성화
#else
			arguments.push_back(L"-O3");        // 최적화 레벨 3
#endif // _DEBUG

			DxcBuffer sourceBuffer = {};
			sourceBuffer.Ptr = sourceBlob->GetBufferPointer();
			sourceBuffer.Size = sourceBlob->GetBufferSize();
			sourceBuffer.Encoding = DXC_CP_ACP;

			// 셰이더 컴파일
			ComPtr<IDxcResult> compileResult;
			hr = dxcCompiler->Compile(&sourceBuffer, arguments.data(), static_cast<UINT32>(arguments.size()), nullptr, IID_PPV_ARGS(&compileResult));
			if (FAILED(hr)) {
				std::cout << "셰이더 컴파일 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// 컴파일 결과 확인
			HRESULT compileHR;
			compileResult->GetStatus(&compileHR);
			if (FAILED(compileHR)) {
				ComPtr<IDxcBlobEncoding> errorBlob;
				compileResult->GetErrorBuffer(&errorBlob);
				if (errorBlob) {
					std::cout << "셰이더 컴파일 오류: " << (char*)errorBlob->GetBufferPointer() << std::endl;
				}
				return false;
			}

			// 컴파일된 셰이더 가져오기
			ComPtr<IDxcBlob> shaderBlob;
			compileResult->GetResult(&shaderBlob);
			if (!shaderBlob) {
				std::cout << "컴파일된 셰이더를 가져올 수 없습니다." << std::endl;
				return false;
			}

			std::cout << "셰이더 컴파일 성공!" << std::endl;

			// 레이트레이싱 파이프라인 상태 객체 생성
			std::vector<D3D12_STATE_SUBOBJECT> subobjects;

			// 1. DXIL 라이브러리
			D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {};
			dxilLibDesc.DXILLibrary.pShaderBytecode = shaderBlob->GetBufferPointer();
			dxilLibDesc.DXILLibrary.BytecodeLength = shaderBlob->GetBufferSize();
			dxilLibDesc.NumExports = 0; // 모든 심볼 내보내기
			dxilLibDesc.pExports = nullptr;

			D3D12_STATE_SUBOBJECT dxilLibSubobject = {};
			dxilLibSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
			dxilLibSubobject.pDesc = &dxilLibDesc;
			subobjects.push_back(dxilLibSubobject);

			// 2. 히트 그룹 (Hit Group) 정의
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

			// 3. 셰이더 구성
			D3D12_RAYTRACING_SHADER_CONFIG shaderConfig = {};
			shaderConfig.MaxPayloadSizeInBytes = 8 * sizeof(float); // float4 크기
			shaderConfig.MaxAttributeSizeInBytes = 2 * sizeof(float); // float2 크기 (barycentrics)

			D3D12_STATE_SUBOBJECT shaderConfigSubobject = {};
			shaderConfigSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
			shaderConfigSubobject.pDesc = &shaderConfig;
			subobjects.push_back(shaderConfigSubobject);

			//// 4. 서브객체와 익스포트 연결 (새로 추가)
			//static const WCHAR* raygenShaderName = L"RaygenShader";
			//static const WCHAR* missShaderName = L"MissShader";

			//static const WCHAR* shaderExports[] = { raygenShaderName, missShaderName, hitGroupName };

			//D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION shaderConfigAssociation = {};
			//shaderConfigAssociation.pSubobjectToAssociate = &shaderConfigSubobject;
			//shaderConfigAssociation.NumExports = 3; // 정확한 개수 지정
			//shaderConfigAssociation.pExports = shaderExports;

			//D3D12_STATE_SUBOBJECT shaderConfigAssociationSubobject = {};
			//shaderConfigAssociationSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
			//shaderConfigAssociationSubobject.pDesc = &shaderConfigAssociation;
			//subobjects.push_back(shaderConfigAssociationSubobject);

			//// 5. 로컬 루트 시그니처 (새로 추가)
			//D3D12_LOCAL_ROOT_SIGNATURE localRootSignature = {};
			//localRootSignature.pLocalRootSignature = m_localRootSignature.Get();

			//D3D12_STATE_SUBOBJECT localRootSignatureSubobject = {};
			//localRootSignatureSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
			//localRootSignatureSubobject.pDesc = &localRootSignature;
			//subobjects.push_back(localRootSignatureSubobject);

			//// 6. 로컬 루트 시그니처 연결
			//D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION localRootSignatureAssociation = {};
			//localRootSignatureAssociation.pSubobjectToAssociate = &localRootSignatureSubobject;
			//localRootSignatureAssociation.NumExports = 3; // 정확한 개수 지정
			//localRootSignatureAssociation.pExports = shaderExports;

			//D3D12_STATE_SUBOBJECT localRootSignatureAssociationSubobject = {};
			//localRootSignatureAssociationSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
			//localRootSignatureAssociationSubobject.pDesc = &localRootSignatureAssociation;
			//subobjects.push_back(localRootSignatureAssociationSubobject);

			// 7. 글로벌 루트 시그니처 연결
			D3D12_GLOBAL_ROOT_SIGNATURE globalRootSignature = {};
			globalRootSignature.pGlobalRootSignature = m_globalRootSignature.Get();

			D3D12_STATE_SUBOBJECT globalRootSignatureSubobject = {};
			globalRootSignatureSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
			globalRootSignatureSubobject.pDesc = &globalRootSignature;
			subobjects.push_back(globalRootSignatureSubobject);

			// 8. 파이프라인 구성
			D3D12_RAYTRACING_PIPELINE_CONFIG pipelineConfig = {};
			pipelineConfig.MaxTraceRecursionDepth = 3; // 재귀 깊이

			D3D12_STATE_SUBOBJECT pipelineConfigSubobject = {};
			pipelineConfigSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
			pipelineConfigSubobject.pDesc = &pipelineConfig;
			subobjects.push_back(pipelineConfigSubobject);

			// 파이프라인 상태 객체 생성
			D3D12_STATE_OBJECT_DESC stateObjectDesc = {};
			stateObjectDesc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
			stateObjectDesc.NumSubobjects = static_cast<UINT>(subobjects.size());
			stateObjectDesc.pSubobjects = subobjects.data();

			hr = m_device->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&m_raytracingPipelineState));
			if (FAILED(hr)) {
				std::cout << "레이트레이싱 파이프라인 상태 객체 생성 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// 상태 객체 속성 가져오기
			hr = m_raytracingPipelineState.As(&m_stateObjectProperties);
			if (FAILED(hr)) {
				std::cout << "상태 객체 속성 가져오기 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			std::cout << "레이트레이싱 파이프라인 생성 성공!" << std::endl;
			return true;

		}
		catch (const std::exception& e) {
			std::cout << "예외 발생: " << e.what() << std::endl;
			return false;
		}
		catch (...) {
			std::cout << "알 수 없는 예외 발생" << std::endl;
			return false;
		}
	}

	// 셰이더 테이블 생성
	bool CreateShaderTables() {
		try {
			std::cout << "셰이더 테이블 생성 중..." << std::endl;

			// 힙 속성
			D3D12_HEAP_PROPERTIES uploadHeapProperties = {};
			uploadHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
			uploadHeapProperties.CreationNodeMask = 1;
			uploadHeapProperties.VisibleNodeMask = 1;

			// Ray Generation 셰이더 테이블 생성
			D3D12_RESOURCE_DESC raygenShaderTableDesc = {
				.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
				.Alignment = 0,
				.Width = kTable, // Ray Generation 셰이더 테이블 크기
				.Height = 1,
				.DepthOrArraySize = 1,
				.MipLevels = 1,
				.Format = DXGI_FORMAT_UNKNOWN,
				.SampleDesc = {.Count = 1, .Quality = 0 }, // 샘플링 없음
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
				std::cout << "Ray Generation 셰이더 테이블 생성 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// Miss 셰이더 테이블 생성
			hr = m_device->CreateCommittedResource(
				&uploadHeapProperties,
				D3D12_HEAP_FLAG_NONE,
				&raygenShaderTableDesc, // 같은 크기 사용
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&m_missShaderTable));

			if (FAILED(hr)) {
				std::cout << "Miss 셰이더 테이블 생성 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// Hit Group 셰이더 테이블 생성
			hr = m_device->CreateCommittedResource(
				&uploadHeapProperties,
				D3D12_HEAP_FLAG_NONE,
				&raygenShaderTableDesc, // 같은 크기 사용
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&m_hitGroupShaderTable));

			if (FAILED(hr)) {
				std::cout << "Hit Group 셰이더 테이블 생성 실패: 0x" << std::hex << hr << std::endl;
				return false;
			}

			// 셰이더 식별자를 테이블에 복사
			void* raygenShaderIdentifier = m_stateObjectProperties->GetShaderIdentifier(L"RaygenShader");
			void* missShaderIdentifier = m_stateObjectProperties->GetShaderIdentifier(L"MissShader");
			void* hitGroupIdentifier = m_stateObjectProperties->GetShaderIdentifier(L"HitGroup");

			if (!raygenShaderIdentifier || !missShaderIdentifier || !hitGroupIdentifier) {
				std::cout << "셰이더 식별자 가져오기 실패" << std::endl;
				return false;
			}

			auto writeRecord = [&](ID3D12Resource* resource, const void* shaderId) -> bool {
				UINT8* pData;
				D3D12_RANGE readRange = {};
				HRESULT hr = resource->Map(0, &readRange, reinterpret_cast<void**>(&pData));
				if (SUCCEEDED(hr)) {
					memcpy(pData, shaderId, kId); // kId는 셰이더 식별자의 크기
					memset(pData + kId, 0, kTable - kId); // 나머지 공간을 0으로 초기화
					resource->Unmap(0, nullptr);
					return true;
				}
				else
				{
					std::cout << "셰이더 테이블 맵핑 실패: 0x" << std::hex << hr << std::endl;
					return false;
				}
				};

			// Ray Generation 셰이더 식별자 복사
			if (!writeRecord(m_raygenShaderTable.Get(), raygenShaderIdentifier)) { return false; }

			// Miss 셰이더 식별자 복사
			if (!writeRecord(m_missShaderTable.Get(), missShaderIdentifier)) { return false; }

			// Hit Group 식별자 복사
			if (!writeRecord(m_hitGroupShaderTable.Get(), hitGroupIdentifier)) { return false; }

			std::cout << "셰이더 테이블 생성 성공!" << std::endl;
			return true;

		}
		catch (...) {
			std::cout << "셰이더 테이블 생성 중 예외 발생" << std::endl;
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
			std::cout << "출력 리소스 생성 실패: 0x" << std::hex << hr << std::endl;
			return false;
		}

		return true;
	}

	bool CreateDescriptors() {
		auto cpuHandle = m_descriptorHeap->GetCPUDescriptorHandleForHeapStart();
		UINT inc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// 0: TLAS SRV 생성(t0)
		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {
			.Format = DXGI_FORMAT_UNKNOWN,
			.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
			.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
			.RaytracingAccelerationStructure = {.Location = m_topLevelAS->GetGPUVirtualAddress() }
		};
		m_device->CreateShaderResourceView(nullptr, &srvDesc, cpuHandle);
		cpuHandle.ptr += inc;

		// 1: CBV 생성(b0)
		D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {
			.BufferLocation = m_constantBuffer->GetGPUVirtualAddress(),
			.SizeInBytes = (sizeof(SceneConstants) + 255) & ~255 // 256바이트 정렬
		};
		m_device->CreateConstantBufferView(&cbvDesc, cpuHandle);
		cpuHandle.ptr += inc;

		// 2: UAV 생성(u0)
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

// 윈도우 프로시저
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	switch (uMsg) {
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	}
	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
	// 윈도우 클래스 등록

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

	// 윈도우 생성
	HWND hwnd = CreateWindowEx(
		0,
		CLASS_NAME,
		L"DXR 정육면체 렌더러",
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, 2560, 1440,
		NULL,
		NULL,
		hInstance,
		NULL
	);

	if (hwnd == NULL) {
		std::cout << "윈도우 생성 실패" << std::endl;
		std::cout << "아무 키나 누르면 종료됩니다..." << std::endl;
		std::cin.get();
		return 0;
	}

	ShowWindow(hwnd, nCmdShow);

	// DXR 렌더러 초기화
	DXRCubeRenderer renderer;
	if (!renderer.Initialize(hwnd, 2560, 1440)) {
		std::cout << "DXR 렌더러 초기화 실패" << std::endl;
		std::cout << "아무 키나 누르면 종료됩니다..." << std::endl;
		std::cin.get();
		return -1;
	}

	std::cout << "DXR 정육면체 렌더러가 시작되었습니다." << std::endl;
	std::cout << "윈도우를 닫으면 프로그램이 종료됩니다." << std::endl;

	// 메시지 루프
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
		// 렌더링
		renderer.Render();
	}

	std::cout << "프로그램이 종료됩니다." << std::endl;
	std::cout << "아무 키나 누르면 콘솔창이 닫힙니다..." << std::endl;
	std::cin.get();

	return 0;
}