# ReMe TaskMemory 가이드

> **Remember Me, Refine Me.** - ReMe는 AI 에이전트를 위한 통합 메모리 시스템입니다.

## 목차

1. [ReMe 메모리 시스템 개요](#1-reme-메모리-시스템-개요)
2. [TaskMemory란?](#2-taskmemory란)
3. [코드 구조 및 주요 컴포넌트](#3-코드-구조-및-주요-컴포넌트)
4. [TaskMemory 호출 방법](#4-taskmemory-호출-방법)
5. [AppWorld Cookbook 적용 사례](#5-appworld-cookbook-적용-사례)
6. [참고 자료](#6-참고-자료)

---

## 1. ReMe 메모리 시스템 개요

ReMe는 AI 에이전트에게 다음 네 가지 유형의 메모리를 제공합니다:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ReMe Memory System                              │
├─────────────────┬─────────────────┬─────────────────┬───────────────────┤
│  Personal       │  Task Memory    │  Tool Memory    │  Working Memory   │
│  Memory         │  (Experience)   │                 │                   │
├─────────────────┼─────────────────┼─────────────────┼───────────────────┤
│ 사용자 선호도   │ 성공/실패 패턴  │ 도구 사용 최적화│ 단기 컨텍스트     │
│ 개인화된 기억   │ 비교 학습       │ 파라미터 최적화 │ 메시지 오프로드   │
│ 시간 인식       │ 검증된 경험     │ 동적 가이드라인 │ 컨텍스트 윈도우   │
└─────────────────┴─────────────────┴─────────────────┴───────────────────┘
```

| 메모리 유형 | 설명 | 주요 용도 |
|------------|------|----------|
| **Personal Memory** | 사용자별 맞춤형 기억 | 사용자 선호도, 상호작용 스타일 이해 |
| **Task Memory** ⭐ | 태스크 실행에서 추출된 경험 | 성공 패턴 인식, 실패 분석, 비교 학습 |
| **Tool Memory** | 도구 사용 패턴 및 최적화 | 성공률 추적, 파라미터 최적화 |
| **Working Memory** | 단기 컨텍스트 메모리 | 메시지 오프로드/리로드, 토큰 관리 |

---

## 2. TaskMemory란?

**TaskMemory(Task Memory/Experience)**는 이전 태스크 실행에서 추출된 지식을 저장하고 재사용하는 메모리 시스템입니다.

### 핵심 특징

- **Success Pattern Recognition**: 효과적인 전략과 그 원리를 식별
- **Failure Analysis Learning**: 실패로부터 학습하여 동일한 실수 방지
- **Comparative Patterns**: 서로 다른 trajectory 비교를 통한 가치 있는 메모리 추출
- **Validation Patterns**: 검증 모듈을 통한 추출된 메모리의 유효성 확인

### TaskMemory 스키마

```python
# reme_ai/schema/memory.py

class TaskMemory(BaseMemory):
    memory_type: str = "task"
    
    # 핵심 필드
    workspace_id: str          # 워크스페이스 식별자
    memory_id: str             # 고유 메모리 ID
    when_to_use: str           # 이 메모리를 사용할 조건
    content: str               # 실제 경험/지식 내용
    score: float               # 관련성 점수 (0.0 ~ 1.0)
    
    # 메타데이터
    time_created: str
    time_modified: str
    author: str
    metadata: dict
```

---

## 3. 코드 구조 및 주요 컴포넌트

### 디렉토리 구조

```
reme_ai/
├── main.py                          # ReMeApp 메인 진입점
├── schema/
│   └── memory.py                    # TaskMemory, PersonalMemory, ToolMemory 스키마
├── service/
│   └── task_memory_service.py       # TaskMemory 서비스 레이어
├── config/
│   └── default.yaml                 # Flow 정의 및 설정
├── summary/
│   └── task/                        # Summary 관련 Operation들
│       ├── trajectory_preprocess_op.py
│       ├── success_extraction_op.py
│       ├── failure_extraction_op.py
│       ├── comparative_extraction_op.py
│       ├── memory_validation_op.py
│       ├── memory_deduplication_op.py
│       └── simple_summary_op.py
└── retrieve/
    └── task/                        # Retrieve 관련 Operation들
        ├── build_query_op.py
        ├── rerank_memory_op.py
        ├── rewrite_memory_op.py
        └── merge_memory_op.py
```

### 주요 Flow 정의

`reme_ai/config/default.yaml`에서 TaskMemory 관련 Flow가 정의됩니다:

```yaml
flow:
  # 메모리 검색 Flow
  retrieve_task_memory:
    flow_content: >
      BuildQueryOp() >> 
      RecallVectorStoreOp() >> 
      RerankMemoryOp(enable_llm_rerank=True, enable_score_filter=True, top_k=5) >> 
      RewriteMemoryOp(enable_llm_rewrite=True)

  # 메모리 요약/저장 Flow  
  summary_task_memory:
    flow_content: >
      TrajectoryPreprocessOp(success_threshold=1.0) >> 
      (SuccessExtractionOp() | FailureExtractionOp() | ComparativeExtractionOp(enable_soft_comparison=True)) >> 
      MemoryValidationOp(validation_threshold=0.5) >> 
      MemoryDeduplicationOp() >> 
      UpdateVectorStoreOp()

  # 메모리 기록 업데이트 Flow
  record_task_memory:
    flow_content: UpdateMemoryFreqOp() >> UpdateMemoryUtilityOp() >> UpdateVectorStoreOp()

  # 메모리 삭제 Flow
  delete_task_memory:
    flow_content: DeleteMemoryOp() >> UpdateVectorStoreOp()
```

### Summary Flow Operations

| Operation | 경로 | 설명 |
|-----------|------|------|
| `TrajectoryPreprocessOp` | `summary/task/trajectory_preprocess_op.py` | Trajectory 전처리 및 성공/실패 분류 |
| `SuccessExtractionOp` | `summary/task/success_extraction_op.py` | 성공 trajectory에서 메모리 추출 |
| `FailureExtractionOp` | `summary/task/failure_extraction_op.py` | 실패 trajectory에서 메모리 추출 |
| `ComparativeExtractionOp` | `summary/task/comparative_extraction_op.py` | 비교 분석을 통한 메모리 추출 |
| `MemoryValidationOp` | `summary/task/memory_validation_op.py` | 추출된 메모리 품질 검증 |
| `MemoryDeduplicationOp` | `summary/task/memory_deduplication_op.py` | 중복 메모리 제거 |

### Retrieve Flow Operations

| Operation | 경로 | 설명 |
|-----------|------|------|
| `BuildQueryOp` | `retrieve/task/build_query_op.py` | 쿼리 생성 (직접 입력 또는 메시지 분석) |
| `RecallVectorStoreOp` | 벡터 스토어 내장 | 벡터 DB에서 관련 메모리 검색 |
| `RerankMemoryOp` | `retrieve/task/rerank_memory_op.py` | LLM 기반 메모리 재순위화 |
| `RewriteMemoryOp` | `retrieve/task/rewrite_memory_op.py` | 컨텍스트에 맞게 메모리 재작성 |
| `MergeMemoryOp` | `retrieve/task/merge_memory_op.py` | 여러 메모리를 단일 응답으로 병합 |

---

## 4. TaskMemory 호출 방법

ReMe 서버는 HTTP API를 통해 TaskMemory 기능을 제공합니다.

### 4.1 서버 시작

```bash
# 기본 실행
reme backend=http http.port=8002

# 또는 Python 모듈로 실행
python -m reme_ai.main backend=http http.port=8002
```

### 4.2 retrieve_task_memory - 메모리 검색

쿼리를 기반으로 관련 태스크 경험을 검색합니다.

**엔드포인트**: `POST /retrieve_task_memory`

**요청 필드**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `workspace_id` | string | ✅ | 워크스페이스 식별자 |
| `query` | string | ✅ | 검색 쿼리 |
| `top_k` | integer | ❌ | 반환할 최대 메모리 수 (기본값: 5) |

**요청 예시**:

```python
import requests

response = requests.post(
    url="http://localhost:8002/retrieve_task_memory",
    json={
        "workspace_id": "appworld_v1",
        "query": "How to handle API authentication errors?",
        "top_k": 5
    }
)

result = response.json()
print(result["answer"])  # 재작성된 경험 텍스트
print(result["metadata"]["memory_list"])  # 원본 메모리 리스트
```

**응답 구조**:

```json
{
    "answer": "Experience 1: When encountering API auth errors...\nExperience 2: ...",
    "metadata": {
        "memory_list": [
            {
                "memory_id": "abc123",
                "when_to_use": "When API returns 401 error",
                "content": "Check token expiration and refresh if needed",
                "score": 0.95
            }
        ]
    }
}
```

### 4.3 summary_task_memory - 메모리 저장

태스크 실행 trajectory에서 경험을 추출하고 저장합니다.

**엔드포인트**: `POST /summary_task_memory`

**요청 필드**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `workspace_id` | string | ✅ | 워크스페이스 식별자 |
| `trajectories` | array | ✅ | 태스크 실행 기록 리스트 |

**Trajectory 구조**:

```python
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        # ... 대화 히스토리
    ],
    "score": 1.0  # 0.0 ~ 1.0 (1.0 = 성공)
}
```

**요청 예시**:

```python
import requests

response = requests.post(
    url="http://localhost:8002/summary_task_memory",
    json={
        "workspace_id": "appworld_v1",
        "trajectories": [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Search for Python tutorials"},
                    {"role": "assistant", "content": "```python\nsearch_api('Python tutorials')\n```"},
                    {"role": "user", "content": "Output: Found 10 results..."}
                ],
                "score": 1.0
            }
        ]
    }
)

result = response.json()
print(f"Created {len(result['metadata']['memory_list'])} memories")
```

### 4.4 record_task_memory - 메모리 정보 업데이트

검색된 메모리의 frequency 및 utility 속성을 업데이트합니다.

**엔드포인트**: `POST /record_task_memory`

**요청 필드**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `workspace_id` | string | ✅ | 워크스페이스 식별자 |
| `memory_dicts` | array | ✅ | 업데이트할 메모리 딕셔너리 리스트 |
| `update_utility` | boolean | ✅ | utility 업데이트 여부 (태스크 성공 시 true) |

**요청 예시**:

```python
import requests

response = requests.post(
    url="http://localhost:8002/record_task_memory",
    json={
        "workspace_id": "appworld_v1",
        "memory_dicts": [
            {"memory_id": "abc123", "when_to_use": "...", "content": "..."}
        ],
        "update_utility": True  # 태스크 성공 시
    }
)
```

### 4.5 delete_task_memory - 저품질 메모리 삭제

utility가 낮고 자주 검색된 메모리를 삭제합니다.

**엔드포인트**: `POST /delete_task_memory`

**요청 필드**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `workspace_id` | string | ✅ | 워크스페이스 식별자 |
| `freq_threshold` | integer | ✅ | 검색 빈도 임계값 |
| `utility_threshold` | float | ✅ | utility/freq 비율 임계값 |

**요청 예시**:

```python
import requests

response = requests.post(
    url="http://localhost:8002/delete_task_memory",
    json={
        "workspace_id": "appworld_v1",
        "freq_threshold": 5,      # 5회 이상 검색된 메모리 대상
        "utility_threshold": 0.5  # utility/freq < 0.5인 메모리 삭제
    }
)
```

### 4.6 vector_store - 벡터 스토어 직접 조작

메모리 라이브러리 로드/덤프, 메모리 삭제 등의 작업을 수행합니다.

**엔드포인트**: `POST /vector_store`

**요청 예시**:

```python
# 메모리 라이브러리 로드
response = requests.post(
    url="http://localhost:8002/vector_store",
    json={
        "workspace_id": "appworld_v1",
        "action": "load",
        "path": "docs/library"
    }
)

# 특정 메모리 ID 삭제
response = requests.post(
    url="http://localhost:8002/vector_store",
    json={
        "workspace_id": "appworld_v1",
        "action": "delete_ids",
        "memory_ids": ["memory_id_1", "memory_id_2"]
    }
)

# 전체 메모리 리스트 조회
response = requests.post(
    url="http://localhost:8002/vector_store",
    json={
        "workspace_id": "appworld_v1",
        "action": "list"
    }
)
```

---

## 5. AppWorld Cookbook 적용 사례

AppWorld는 복잡한 태스크 계획 및 실행 패턴을 다루는 벤치마크입니다. `cookbook/appworld` 예제에서 TaskMemory가 어떻게 활용되는지 살펴봅니다.

### 5.1 작동 흐름

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        AppWorld + TaskMemory 작동 흐름                           │
└─────────────────────────────────────────────────────────────────────────────────┘

1. 태스크 시작
   ┌──────────────┐
   │  Task Start  │
   │  (instruction)│
   └──────┬───────┘
          │
          ▼
2. 메모리 검색 ─────────────────────────────────────────────────────────────────────
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  retrieve_task_memory                                                        │
   │  ┌────────────┐   ┌─────────────────┐   ┌───────────────┐   ┌─────────────┐ │
   │  │ BuildQuery │ → │ RecallVectorStore│ → │ RerankMemory │ → │RewriteMemory│ │
   │  │    Op      │   │       Op         │   │      Op      │   │     Op      │ │
   │  └────────────┘   └─────────────────┘   └───────────────┘   └─────────────┘ │
   └──────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
3. 프롬프트 구성
   ┌────────────────────────────────────────────────┐
   │  Task: {instruction}                           │
   │                                                │
   │  Some Related Experience to help you:          │
   │  Experience 1: When to use: ...                │
   │               Content: ...                     │
   │  Experience 2: ...                             │
   └────────────────────────────────────────────────┘
          │
          ▼
4. ReAct 에이전트 실행 (LLM 호출 → 코드 실행 반복)
   ┌──────────────────────────────────────────────┐
   │  for i in range(max_interactions):           │
   │      code = call_llm(messages)               │
   │      output = world.execute(code)            │
   │      if task_completed: break                │
   └──────────────────────────────────────────────┘
          │
          ▼
5. 태스크 완료 후 처리 ──────────────────────────────────────────────────────────────
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  성공 시:                                                                    │
   │  ┌───────────────────────┐   ┌────────────────────────┐                     │
   │  │ summary_task_memory   │   │ record_task_memory     │                     │
   │  │ (경험 추출 & 저장)     │   │ (utility 업데이트)      │                     │
   │  └───────────────────────┘   └────────────────────────┘                     │
   │                                                                              │
   │  실패 시 (Failure-Aware Reflection):                                         │
   │  ┌───────────────────────┐   ┌────────────────────────┐                     │
   │  │ summary_task_memory   │ → │ 다음 시도에서 사용      │                     │
   │  │ (실패 경험 추출)       │   │ (previous_memories)    │                     │
   │  └───────────────────────┘   └────────────────────────┘                     │
   │                                                                              │
   │  주기적 정리:                                                                │
   │  ┌───────────────────────┐                                                  │
   │  │ delete_task_memory    │                                                  │
   │  │ (저품질 메모리 삭제)    │                                                  │
   │  └───────────────────────┘                                                  │
   └──────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 AppworldReactAgent 코드 분석

`cookbook/appworld/appworld_react_agent.py`에서 TaskMemory 사용 예시:

#### 메모리 검색 (get_memory)

```python
def get_memory(self, query: str):
    """Retrieve relevant task memories based on a query"""
    response = requests.post(
        url=f"{self.memory_base_url}retrieve_task_memory",
        json={
            "workspace_id": self.memory_workspace_id,
            "query": query,
        },
    )
    return response.json()
```

#### 메모리 저장 (add_memory)

```python
def add_memory(self, trajectories):
    """Generate a summary of conversation messages and create task memories"""
    response = requests.post(
        url=f"{self.memory_base_url}summary_task_memory",
        json={
            "workspace_id": self.memory_workspace_id,
            "trajectories": trajectories,
        },
    )
    return response.json().get("metadata", {}).get("memory_list", [])
```

#### 메모리 정보 업데이트 (update_memory_information)

```python
def update_memory_information(self, memory_list, update_utility: bool = False):
    response = requests.post(
        url=f"{self.memory_base_url}record_task_memory",
        json={
            "workspace_id": self.memory_workspace_id,
            "memory_dicts": memory_list,
            "update_utility": update_utility,
        },
    )
```

#### 저품질 메모리 삭제 (delete_memory)

```python
def delete_memory(self):
    response = requests.post(
        url=f"{self.memory_base_url}delete_task_memory",
        json={
            "workspace_id": self.memory_workspace_id,
            "freq_threshold": self.freq_threshold,
            "utility_threshold": self.utility_threshold,
        },
    )
```

### 5.3 프롬프트에 경험 통합

검색된 메모리를 태스크 프롬프트에 통합하는 방식:

```python
def prompt_messages(self, run_id, task_index, previous_memories, world):
    query = world.task.instruction
    
    if self.use_memory:
        if len(previous_memories) == 0:
            # 벡터 스토어에서 관련 경험 검색
            response = self.get_memory(world.task.instruction)
            task_memory = response["answer"]
            
            # 프롬프트에 경험 추가
            query = (
                "Task:\n" + query + 
                "\n\nSome Related Experience to help you to complete the task:\n" + 
                task_memory
            )
        else:
            # 이전 실패에서 추출된 경험 사용 (Failure-Aware Reflection)
            formatted_memories = []
            for i, memory in enumerate(previous_memories, 1):
                memory_text = f"Experience {i}:\n When to use: {memory['when_to_use']}\n Content: {memory['content']}\n"
                formatted_memories.append(memory_text)
            
            query = (
                "Task:\n" + query + 
                "\n\nSome Related Experience to help you to complete the task:\n" + 
                "\n".join(formatted_memories)
            )
```

### 5.4 Self-Evolving Memory 데모 실행

```bash
cd cookbook/appworld

# 비교 모드: 메모리 사용/미사용 비교
python demo_single_task.py --task-id 3d9a636_1 --compare

# 메모리 사용 모드만
python demo_single_task.py --task-id 3d9a636_1 --with-memory-only

# Failure-Aware Reflection (실패 시 재시도)
python demo_single_task.py --task-id 3d9a636_1 --with-memory-only --num-trials 3
```

---

## 6. 참고 자료

### 공식 문서

- [ReMe 공식 문서](https://reme.agentscope.io/index.html)
- [Task Memory 가이드](https://reme.agentscope.io/task_memory/task_memory.html)
- [Task Memory Retrieval Ops](https://reme.agentscope.io/task_memory/task_retrieve_ops.html)
- [Task Memory Summary Ops](https://reme.agentscope.io/task_memory/task_summary_ops.html)

### GitHub 레포지토리

- [ReMe GitHub](https://github.com/agentscope-ai/ReMe)

### 관련 파일 경로

| 파일/디렉토리 | 설명 |
|--------------|------|
| `reme_ai/main.py` | ReMeApp 메인 클래스 |
| `reme_ai/service/task_memory_service.py` | TaskMemory 서비스 레이어 |
| `reme_ai/schema/memory.py` | 메모리 스키마 정의 |
| `reme_ai/config/default.yaml` | Flow 및 설정 정의 |
| `reme_ai/summary/task/` | Summary 관련 Operations |
| `reme_ai/retrieve/task/` | Retrieve 관련 Operations |
| `cookbook/appworld/` | AppWorld 예제 코드 |

---

## 빠른 시작 예제

```python
import requests

BASE_URL = "http://localhost:8002"
WORKSPACE_ID = "my_workspace"

# 1. 메모리 검색
def retrieve_memory(query: str):
    response = requests.post(
        f"{BASE_URL}/retrieve_task_memory",
        json={"workspace_id": WORKSPACE_ID, "query": query}
    )
    return response.json()

# 2. 메모리 저장
def save_memory(messages: list, score: float):
    response = requests.post(
        f"{BASE_URL}/summary_task_memory",
        json={
            "workspace_id": WORKSPACE_ID,
            "trajectories": [{"messages": messages, "score": score}]
        }
    )
    return response.json()

# 3. 사용 예시
# 태스크 시작 전 관련 경험 검색
experiences = retrieve_memory("How to handle file upload errors?")
print(experiences["answer"])

# 태스크 완료 후 경험 저장
task_history = [
    {"role": "user", "content": "Upload a file to S3"},
    {"role": "assistant", "content": "Using boto3 to upload..."},
    {"role": "user", "content": "Success!"}
]
save_memory(task_history, score=1.0)
```

---

*이 문서는 ReMe 프로젝트의 TaskMemory 시스템을 설명합니다. 자세한 내용은 [공식 문서](https://reme.agentscope.io)를 참조하세요.*
