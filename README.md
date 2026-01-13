# NIKA with ReMe

NIKA 벤치마크를 ReMe 메모리 시스템으로 평가하기 위한 리포지토리입니다.

## 1. 환경 설정 (uv 사용)

NIKA와 ReMe는 서로 다른 가상환경을 사용합니다. 각각의 디렉토리에서 가상환경을 설정해주세요.

### 1-1. ReMe 환경 설정 (ReMe 서버 실행용)

ReMe 루트 디렉토리(`ReMe/`)에서 진행합니다.

```bash
# ReMe 루트로 이동
# 가상환경 생성 (.venv)
uv venv .venv-reme

# 패키지 설치
uv pip install -r requirements-reme.txt
```

### 1-2. NIKA 환경 설정 (벤치마크 실행용)

NIKA 디렉토리(`ReMe/cookbook/nika/`)에서 진행합니다.

```bash
# NIKA 디렉토리로 이동
cd cookbook/nika

# 가상환경 생성 (이름은 .venv-nika 권장)
uv venv .venv-nika

# NIKA 패키지 설치
uv pip install -r requirements-nika.txt
```

## 2. ReMe 서버 실행

ReMe 메모리 서버를 실행합니다. **(새 터미널에서 실행 권장)**

```bash
# ReMe 루트 디렉토리에서 실행
# ReMe 가상환경 활성화
source .venv/bin/activate

# 서버 실행
reme serve vector_store.default.backend=memory

# 임베딩 값 로컬에 json에 저장 및 로딩해서 사용하고 싶으면
reme serve vector_store.default.backend=local
```

## 3. 벤치마크 실행

ReMe 메모리를 활성화하여 벤치마크를 실행합니다. **(다른 터미널에서 실행)**

```bash
# NIKA 디렉토리(ReMe/cookbook/nika)에서 실행
cd cookbook/nika

# NIKA 가상환경 활성화
source .venv-nika/bin/activate

# 벤치마크 실행
# --use-memory: ReMe 메모리 사용 활성화
# --num-trials 2: 실패 시 최대 2번까지 재시도 (이전 실패 경험을 메모리로 활용/Failure-aware Reflection)
python benchmark/run_benchmark.py --use-memory --num-trials 2
```