import os
import json
import pandas as pd
import google.generativeai as genai

# =====================================================================
# 1. 기존 추천 시스템 클래스 불러오기 (이전 코드에서 만든 클래스)
# =====================================================================
# from recommender import ProductionCrossMediaRecommender
# 여기서는 코드가 길어지므로 recommender 인스턴스가 있다고 가정합니다.
recommender = ProductionCrossMediaRecommender()

# =====================================================================
# 2. DB / 데이터 패치 모의 클래스 (실제 서비스 시 CSV나 DB 쿼리로 교체)
# =====================================================================
class DataStore:
    @staticmethod
    def get_item_by_title(item_type: str, title: str) -> pd.Series:
        """입력받은 제목으로 해당 작품의 데이터를 가져옵니다."""
        # 실제 환경: SELECT * FROM webtoons WHERE series_title = title
        # 아래는 테스트용 Mock 데이터입니다.
        if item_type == "webtoon":
            return pd.Series({"series_title": title, "mood_eng": "dark", "genres": "액션, 스릴러"})
        elif item_type == "movie":
            return pd.Series({"title": title, "poster_mood_eng": "romantic", "runtime": 120})
        elif item_type == "music":
            return pd.Series({"track_name": title, "energy": 0.8, "valence": 0.3})
        return pd.Series()

    @staticmethod
    def get_candidates(item_type: str) -> pd.DataFrame:
        """추천의 대상이 될 후보군 100개를 가져옵니다."""
        # 실제 환경: DB에서 인기도순 100개 or 랜덤 100개 추출
        return pd.DataFrame() # 실제 데이터프레임 리턴

# =====================================================================
# 3. Gemini가 사용할 도구(Tool) 정의 - 💡 이 함수의 Docstring이 매우 중요!
# =====================================================================
def get_cross_media_recommendations(
    source_type: str, 
    source_title: str, 
    target_type: str, 
    top_k: int = 3
) -> str:
    """
    사용자가 특정 미디어(웹툰, 영화, 음악)를 언급했을 때, 다른 미디어(웹툰, 영화, 음악)를 추천해주는 함수입니다.
    
    Args:
        source_type: 기준이 되는 작품의 형태 ('webtoon', 'movie', 'music' 중 하나)
        source_title: 기준이 되는 작품의 제목 (예: "나 혼자만 레벨업", "인터스텔라", "Ditto")
        target_type: 추천받고 싶은 작품의 형태 ('webtoon', 'movie', 'music' 중 하나)
        top_k: 추천할 작품의 개수 (기본값 3)
        
    Returns:
        추천 결과 리스트를 JSON 문자열 형식으로 반환합니다.
    """
    print(f"⚙️ [System] 추천 알고리즘 가동 중... ({source_title} -> {target_type} 추천 중)")
    
    # 1. 입력받은 작품의 데이터 가져오기
    source_item = DataStore.get_item_by_title(source_type, source_title)
    if source_item.empty:
        return json.dumps({"error": f"데이터베이스에서 '{source_title}'을(를) 찾을 수 없습니다."})

    # 2. 추천 대상 후보군 가져오기
    candidate_df = DataStore.get_candidates(target_type)
    if candidate_df.empty:
        # ⚠️ 테스트 환경용 가짜 응답 세팅 (실제 환경에선 지우세요)
        return json.dumps([
            {"title": f"추천된 {target_type} 1", "final_score": 0.85, "reason_keyword": "Dopamine Track Match"},
            {"title": f"추천된 {target_type} 2", "final_score": 0.78, "reason_keyword": "Mood Similarity"}
        ])

    # 3. 추천 로직 실행 (이전에 만든 클래스 호출)
    recommended_df = recommender.recommend(
        source_item=source_item,
        source_type=source_type,
        candidate_df=candidate_df,
        target_type=target_type,
        alpha=0.6,
        top_k=top_k
    )
    
    # 4. LLM이 읽기 쉽게 JSON으로 변환하여 리턴
    # 대상이 영화면 title, 음악이면 track_name, 웹툰이면 series_title을 추출하도록 전처리
    title_col = "title" if target_type == "movie" else "track_name" if target_type == "music" else "series_title"
    
    results = recommended_df[[title_col, "final_score", "mood_score"]].to_dict(orient="records")
    return json.dumps(results)

# =====================================================================
# 4. Gemini Agent 챗봇 설정 및 실행
# =====================================================================

# API 키 설정 (본인의 API 키로 변경)
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# 에이전트 페르소나(시스템 프롬프트) 부여
system_instruction = """
너는 크로스 미디어(웹툰, 영화, 음악)를 넘나들며 사용자 취향에 맞는 작품을 큐레이션 해주는 전문 추천 AI 에이전트야.
너는 단순히 비슷한 장르를 추천하는 것을 넘어, 다음의 EDA 데이터 인사이트를 기반으로 추천 결과를 사용자에게 설명해야 해:
1. 유저의 심리 상태(도파민 스파이크형 소비 vs 세로토닌 힐링형 소비)를 파악하고 그에 맞춰 연결해.
2. 음악 추천 시 '3분 법칙'과 대중적인 코드가 어떻게 반영되었는지 설명해.
3. 영화 추천 시 생존 편향이 덜한 최신 텐트폴 영화인지, 평점이 보장된 웰메이드 영화인지 언급해.

사용자가 특정 작품을 말하며 추천을 요구하면, 반드시 `get_cross_media_recommendations` 함수를 호출하여 데이터를 가져온 뒤, 그 결과를 바탕으로 전문적이고 친절하게 설명해줘.
"""

# 모델 인스턴스 생성 (함수 도구 장착)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  # 추론 속도를 위해 flash 권장
    tools=[get_cross_media_recommendations],
    system_instruction=system_instruction
)

# 챗봇 세션 시작 (대화 기록 유지)
chat = model.start_chat(enable_automatic_function_calling=True)

# =====================================================================
# 5. 실제 채팅 테스트 (Terminal CLI)
# =====================================================================
print("🤖 추천 에이전트가 준비되었습니다. (종료하려면 'exit' 입력)")
while True:
    user_input = input("\n🧑‍💻 사용자: ")
    if user_input.lower() in ['exit', 'quit']:
        break
        
    # Gemini에게 메시지 전송
    response = chat.send_message(user_input)
    print(f"\n🤖 에이전트:\n{response.text}")
