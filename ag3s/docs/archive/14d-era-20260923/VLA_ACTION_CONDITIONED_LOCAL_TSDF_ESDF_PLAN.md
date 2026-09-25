# VLA Action-Conditioned Local TSDF/ESDF Collision Pipeline 개발 계획

## 1. 배경 및 목적

현재 VLA가 생성한 **action chunk를 실행하기 전에 충돌 가능성을 검사**하고, 필요한 경우 Trajectory Optimization(TO)을 통해 충돌이 없고 부드러운 궤적으로 보정하는 구조를 개발한다.

본 구조는 **cuRoboV2의 TSDF/ESDF perception pipeline**을 참고한다.

cuRoboV2는 다음 구조를 사용한다.

- 환경 표현: **Fine-resolution Block-Sparse TSDF**
- Collision query: **Dense ESDF**
- TSDF와 ESDF 해상도 분리
  - TSDF: 정밀한 geometry reconstruction
  - ESDF: robot collision checking에 필요한 task-specific resolution
- Dense ESDF를 이용한 빠른 distance query
- Sparse TSDF → Dense ESDF 변환 시 gather 기반 seeding 및 PBA+ 사용

즉, cuRoboV2는 perception용 TSDF와 planning용 ESDF의 역할을 분리한다.

본 개발에서는 이 구조를 참고하되, **전체 workspace에 Dense ESDF를 생성하는 부분을 VLA에 특화된 Local Dense ESDF 방식으로 변경**한다.

---

## 2. 제안 구조

cuRoboV2의

```text
Fine Block-Sparse TSDF
        ↓
Full Workspace Dense ESDF
        ↓
Trajectory Optimization
```

구조를 다음과 같이 변경한다.

```text
Depth Observation
        ↓
Fine Block-Sparse TSDF
        ↓
VLA Action Chunk
        ↓
Forward Kinematics
        ↓
Robot Collision Sphere Trajectories
        ↓
Swept Volume + Safety Margin
        ↓
VLA Action-Conditioned Local ROI
        ↓
Local Dense ESDF
        ↓
Collision Distance / Gradient Query
        ↓
Trajectory Optimization
        ↓
Safe & Smooth Action Chunk
```

핵심은 다음과 같다.

> **전체 공간에서는 Sparse TSDF를 유지하고, 현재 VLA action chunk가 실제로 통과할 것으로 예상되는 영역에 대해서만 Dense ESDF를 생성한다.**

따라서 본 방식은 nvblox처럼 ESDF 자체를 sparse block 구조로 유지하는 방식과는 다르다.

```text
Sparse in global workspace
+
Dense inside VLA-conditioned local ROI
```

즉, cuRoboV2의 **Full Workspace Dense ESDF**를

```text
VLA Action-Conditioned Local Dense ESDF
```

로 대체한다.

---

## 3. 구현 핵심

1. Depth observation을 이용한 **Block-Sparse TSDF** 환경 표현을 구성한다.
2. VLA가 출력한 action chunk에 대해 Forward Kinematics를 수행한다.
3. 각 timestep에서 robot collision sphere의 위치를 계산한다.
4. 전체 action chunk에 대한 collision sphere의 **swept volume**을 계산한다.
5. Swept volume에 configurable **safety margin / padding**을 적용하여 Local ROI를 결정한다.
6. 전체 workspace가 아니라 해당 ROI에 대해서만 **Dense ESDF**를 생성한다.
7. TSDF와 ESDF resolution은 독립적으로 설정할 수 있도록 한다.

예:

```text
TSDF voxel size: 5 mm
ESDF voxel size: 10~20 mm
```

8. Local Dense ESDF에서 TO가 반복적으로 사용할 수 있도록 다음 query를 제공한다.

```text
signed distance d(p)
distance gradient ∇d(p)
```

9. TO 과정에서 trajectory가 초기 ROI 밖으로 이동할 가능성을 고려하여 ROI padding을 충분히 확보하거나 필요 시 ROI를 갱신한다.

---

## 4. 개발 및 실행 환경

현재 주요 개발 및 연산 공간은 **GPU Server**이다.

단, `pi05_infer.py`는 **로컬 PC에서 실행**한다.

### 접속 구조

로컬 PC에서 GPU Server 접속:

```bash
ssh blunex@ai.amrc.kr -p 21151
```

GPU Server에서 작업 컨테이너 접속:

```bash
ssh root@172.21.121.112 -p 32542
```

개발 및 GPU 연산은 기본적으로 컨테이너 내부에서 수행한다.

```text
Local PC
 └─ pi05_infer.py

GPU Server
 └─ Container
     ├─ TSDF
     ├─ ESDF
     ├─ FK / Collision
     └─ Trajectory Optimization
```

---

## 5. 코드 작업 위치

구현 위치는 현재 프로젝트 구조를 확인한 뒤 선택한다.

가능한 방법:

```text
1. 기존 ag3s package 내부에서 구현
```

또는

```text
2. 새로운 package / folder를 생성하여 구현
```

기존 구조와 의존성을 먼저 확인하고, 기존 `ag3s`와의 결합도가 높은 경우 `ag3s` 내부에서 개발한다. 독립적인 perception / collision pipeline으로 관리하는 것이 더 적합한 경우 새로운 폴더를 생성한다.

---

## 6. 테스트 방법

구현 결과는 먼저 기존 데이터 또는 실제 VLA inference를 통해 검증한다.

### Test A. run0004 기반 테스트

가능한 경우 기존 `run0004` 데이터를 이용하여 다음을 확인한다.

- VLA action chunk 입력
- FK 결과
- collision sphere trajectory
- swept volume
- Local ROI
- TSDF / ESDF 생성 결과
- action chunk의 collision 여부
- TO 적용 전/후 trajectory 비교

### Test B. 실제 모델 기반 테스트

실제 모델을 사용하는 경우 다음 checkpoint를 사용한다.

```text
checkpoint: 29999
```

실제 VLA inference는 로컬의 `pi05_infer.py`에서 수행하고, 생성된 action chunk를 GPU Server의 collision / TO pipeline으로 전달하여 검증한다.

---

## 7. 개발 진행 방식

개발을 시작할 때 **진행 상황 기록용 Markdown 파일을 하나 생성**하고, 이후 작업 내용을 계속 누적 기록한다.

예:

```text
docs/VLA_LOCAL_ESDF_PROGRESS.md
```

최소한 다음 항목을 계속 기록한다.

```text
- 날짜
- 수행 내용
- 수정한 파일
- 실행 명령
- 사용한 parameter
- 테스트 데이터
- 성공/실패 결과
- 문제점
- 다음 작업
```

예시:

```md
## 2026-09-07

### 수행
- VLA action chunk FK 변환 구현
- collision sphere trajectory 생성

### 테스트
- run0004

### 결과
- FK 정상 확인
- swept volume ROI 생성 확인

### 문제
- ROI padding이 작을 경우 TO iteration 중 일부 sphere가 ROI 밖으로 이동

### Next
- adaptive ROI padding 구현
```

---

## 8. Plan 관리

개발 전 작성한 **plan 문서는 `docs/` 디렉터리에 유지**한다.

개발 중에는:

```text
docs/
├─ <기존 plan 문서>
└─ VLA_LOCAL_ESDF_PROGRESS.md
```

형태로 관리하며,

- `plan` 문서: 전체 개발 방향 및 설계
- `progress` 문서: 실제 구현 및 테스트 진행 상황

으로 역할을 분리한다.

---

## 9. 우선 개발 순서

```text
Step 1. 현재 프로젝트 / ag3s 구조 확인

Step 2. 기존 depth / point cloud 입력 경로 확인

Step 3. Block-Sparse TSDF 생성

Step 4. VLA action chunk → FK

Step 5. Robot collision sphere trajectory 생성

Step 6. Swept volume + Local ROI 생성

Step 7. Local Dense ESDF 생성

Step 8. ESDF distance / gradient query 구현

Step 9. 기존 TO와 연결

Step 10. run0004 테스트

Step 11. checkpoint 29999 실제 inference 테스트

Step 12. 성능 평가
         - ESDF 생성 시간
         - collision query 시간
         - TO 시간
         - 전체 latency
         - collision avoidance 성공 여부
```

---

## 최종 목표

최종적으로 다음 pipeline을 구성한다.

```text
VLA Action Chunk
        ↓
Action-Conditioned Collision Region Prediction
        ↓
Local TSDF/ESDF Collision Representation
        ↓
Trajectory Optimization
        ↓
Collision-Free & Smooth Action Chunk
```

핵심 연구 아이디어는 다음과 같다.

> **cuRoboV2의 Fine Sparse TSDF → Dense ESDF 구조를 기반으로 하되, Dense ESDF 생성 영역을 전체 workspace가 아니라 VLA action chunk의 predicted swept volume 주변으로 동적으로 제한한다.**
