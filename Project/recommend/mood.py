
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class ProductionCrossMediaRecommender:

    def __init__(self):
        self.audio_features = [
            "energy",
            "tempo",
            "danceability",
            "loudness",
            "speechiness",
            "acousticness",
            "valence",
        ]
        self.scaler = MinMaxScaler()

        # [EDA 반영] 실제 분석하신 장르별 수치 기반 오디오 프로필 세팅
        self.mood_to_audio_profile = {
            "cheerful": {
                "energy": 0.7,
                "tempo": 115.0,
                "danceability": 0.7,
                "loudness": -5.0,
                "speechiness": 0.07,
                "acousticness": 0.2,
                "valence": 0.8,
            },
            "dark": {
                "energy": 0.4,
                "tempo": 85.0,
                "danceability": 0.4,
                "loudness": -9.0,
                "speechiness": 0.15,
                "acousticness": 0.5,
                "valence": 0.1,
            },
            "thrilling": {
                "energy": 0.85,
                "tempo": 130.0,
                "danceability": 0.6,
                "loudness": -4.0,
                "speechiness": 0.12,
                "acousticness": 0.1,
                "valence": 0.3,
            },
            "romantic": {
                "energy": 0.5,
                "tempo": 90.0,
                "danceability": 0.5,
                "loudness": -7.0,
                "speechiness": 0.06,
                "acousticness": 0.4,
                "valence": 0.6,
            },
            "default": {
                "energy": 0.6,
                "tempo": 105.0,
                "danceability": 0.6,
                "loudness": -6.0,
                "speechiness": 0.08,
                "acousticness": 0.3,
                "valence": 0.5,
            },
        }

    def _classify_webtoon_track(self, item: pd.Series) -> str:
        """[EDA 반영] 웹툰을 도파민형과 세로토닌형 트랙으로 분류"""
        genres = str(item.get("genres", "")).fillna("")
        mood = str(item.get("mood_eng", "")).lower()

        dopamine_keywords = ["액션", "스릴러", "SF", "미래", "판타지", "dark", "thrilling"]
        serotonin_keywords = ["힐링", "코미디", "일상", "cheerful", "romantic"]

        if any(k in genres or k in mood for k in dopamine_keywords):
            return "Dopamine"
        if any(k in genres or k in mood for k in serotonin_keywords):
            return "Serotonin"
        return "Standard"

    def _calculate_music_duration_score(self, duration_ms: float) -> float:
        """[EDA 반영] 3분(180,000ms) 법칙 비대칭 가중치 계산 함수"""
        optimal = 180000.0  # 3분
        diff = duration_ms - optimal

        if diff < 0:
            # 3분보다 짧음: 완만하게 감소
            score = np.exp(-((diff / 60000.0) ** 2))
        else:
            # 3분보다 길음: 급격하게 감소 (도파민 한계 절벽)
            score = np.exp(-((diff / 30000.0) ** 2))
        return float(score)

    def _calculate_music_title_bonus(self, track_name: str) -> float:
        """[EDA 반영] 직관적 단어 수(3~4개) 및 피처링 대중성 보너스"""
        if not isinstance(track_name, str):
            return 0.0
        words = track_name.split()
        bonus = 0.0

        # 직관적인 타이틀 단어 수 선호 (3~4개 단어)
        if 3 <= len(words) <= 4:
            bonus += 0.05

        # 치트키들의 협업 (Feat. / with) 포함 시 인지도 보너스
        if "feat" in track_name.lower() or "with" in track_name.lower():
            bonus += 0.05
        return bonus

    def _calculate_movie_bayesian_rating(
        self, df: pd.DataFrame, m: float = 5000.0
    ) -> pd.Series:
        """[EDA 반영] 생존 편향 및 평점 불균형을 해결하기 위한 베이지안 평균 평점 산출"""
        C = df["vote_average"].mean()  # 전체 영화의 평균 평점
        v = df["vote_count"].fillna(0)
        R = df["vote_average"].fillna(0)

        # 베이지안 평점 공식 적용
        bayesian_avg = (v * R + m * C) / (v + m)
        return bayesian_avg

    def recommend(
        self,
        source_item: pd.Series,
        source_type: str,
        candidate_df: pd.DataFrame,
        target_type: str,
        alpha: float = 0.6,
        top_k: int = 5,
    ) -> pd.DataFrame:

        df = candidate_df.copy()
        if df.empty:
            return df

        # 소스 아이템의 소비 트랙 성향 파악
        source_track = (
            self._classify_webtoon_track(source_item)
            if source_type == "webtoon"
            else "Standard"
        )

        # ==========================================
        # STAGE 2. MOOD MATCHING & COGNITIVE FIT
        # ==========================================
        # [기존 로직 유지하되 EDA 수치 맵핑 활용]
        if source_type in ["webtoon", "movie"] and target_type in [
            "webtoon",
            "movie",
        ]:
            src_col = (
                "mood_eng" if source_type == "webtoon" else "poster_mood_eng"
            )
            tgt_col = (
                "mood_eng" if target_type == "webtoon" else "poster_mood_eng"
            )

            tfidf = TfidfVectorizer()
            all_moods = df[tgt_col].fillna("").tolist() + [source_item[src_col]]
            matrix = tfidf.fit_transform(all_moods)
            df["mood_score"] = cosine_similarity(matrix[:-1], matrix[-1]).flatten()

        elif source_type == "music" and target_type == "music":
            scaled_candidates = self.scaler.fit_transform(
                df[self.audio_features].fillna(0)
            )
            scaled_src = self.scaler.transform(
                source_item[self.audio_features].values.reshape(1, -1)
            )
            df["mood_score"] = cosine_similarity(
                scaled_candidates, scaled_src
            ).flatten()

        else:
            # Text <-> Audio 교차 매칭
            if target_type == "music":
                src_col = (
                    "mood_eng"
                    if source_type == "webtoon"
                    else "poster_mood_eng"
                )
                mood_key = str(source_item[src_col]).lower().strip()
                profile = self.mood_to_audio_profile.get(
                    mood_key, self.mood_to_audio_profile["default"]
                )
                target_vector = np.array([profile[f] for f in self.audio_features]).reshape(1, -1)

                scaled_candidates = self.scaler.fit_transform(df[self.audio_features].fillna(0))
                scaled_target = self.scaler.transform(target_vector)
                df["mood_score"] = cosine_similarity(scaled_candidates, scaled_target).flatten()
            else:
                tgt_col = (
                    "mood_eng"
                    if target_type == "webtoon"
                    else "poster_mood_eng"
                )
                scaled_src = self.scaler.fit_transform(source_item[self.audio_features].values.reshape(1, -1))

                candidate_vectors = []
                for mood in df[tgt_col]:
                    mood_key = str(mood).lower().strip()
                    profile = self.mood_to_audio_profile.get(mood_key, self.mood_to_audio_profile["default"])
                    candidate_vectors.append([profile[f] for f in self.audio_features])

                scaled_candidates = self.scaler.transform(np.array(candidate_vectors))
                df["mood_score"] = cosine_similarity(scaled_candidates, scaled_src).flatten()

        # ==========================================
        # STAGE 3. RANKING & COGNITIVE BIAS CORRECTION (EDA 기반 핵심 고도화)
        # ==========================================

        # 1. 대상이 [음악]일 때의 랭킹 가중치 고도화
        if target_type == "music":
            df["norm_pop"] = self.scaler.fit_transform(df[["track_popularity"]].fillna(0))
            
            # EDA 인사이트 1: 3분 법칙 점수 계산
            df["duration_score"] = df["duration_ms"].apply(self._calculate_music_duration_score)
            
            # EDA 인사이트 2: 타이틀 구조 보너스 계산
            df["title_bonus"] = df["track_name"].apply(self._calculate_music_title_bonus)

            # 음악의 최종 대중성 점수 결합
            df["pop_score"] = (df["norm_pop"] * 0.6) + (df["duration_score"] * 0.3) + df["title_bonus"]

        # 2. 대상이 [영화]일 때의 랭킹 가중치 고도화
        elif target_type == "movie":
            df["norm_pop"] = self.scaler.fit_transform(df[["popularity"]].fillna(0))
            
            # EDA 인사이트 1: 베이지안 평점으로 생존 편향 및 평점 왜곡 보정
            df["bayesian_rating"] = self._calculate_movie_bayesian_rating(df)
            df["norm_rating"] = self.scaler.fit_transform(df[["bayesian_rating"]])

            # EDA 인사이트 2: 개봉 월별 시기성 지표 추출 및 반영 (4,6월=텐트폴, 8,9월=웰메이드)
            df["release_month"] = pd.to_datetime(df["release_date"], errors="coerce").dt.month.fillna(0).astype(int)
            
            # 기본 결합 점수 (인기도 50% + 평점 50%)
            df["pop_score"] = (df["norm_pop"] * 0.5) + (df["norm_rating"] * 0.5)

            # 월별 보정치 적용
            df.loc[df["release_month"].isin([4, 6]), "pop_score"] += 0.05  # 대형 텐트폴 보너스
            df.loc[df["release_month"].isin([8, 9]), "pop_score"] += 0.05  # 완성도 명작 보너스

            # EDA 인사이트 3: 빠른 도파민형 유저를 위한 최신 영화 런타임 가중치 보정
            if source_track == "Dopamine":
                # 런타임이 90~110분 사이인 영화에 최적화 가중치 부여
                df.loc[(df["runtime"] >= 90) & (df["runtime"] <= 110), "pop_score"] += 0.05

        # 3. 대상이 [웹툰]일 때의 랭킹 가중치 고도화
        elif target_type == "webtoon":
            # 로그 스케일 적용하여 sales_point 최상위 솔림 현상 방지
            df["log_sales"] = np.log1p(df["sales_point"].fillna(0))
            df["pop_score"] = self.scaler.fit_transform(df[["log_sales"]])

            # EDA 인사이트 1: 소비 트랙 일치 시 시너지 보너스
            df["target_track"] = df.apply(self._classify_webtoon_track, axis=1)
            if source_track != "Standard":
                df.loc[df["target_track"] == source_track, "pop_score"] += 0.1

        # 최종 스코어 결합 및 정렬
        df["final_score"] = (alpha * df["mood_score"]) + ((1 - alpha) * df["pop_score"])
        return df.sort_values(by="final_score", ascending=False).head(top_k)
