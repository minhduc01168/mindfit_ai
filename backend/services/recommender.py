"""
MindFit AI - Personalized Exercise Recommender Engine
Content-Based Recommendation with Physiological State Matching
Matches user's current psychological/physiological state with sports medicine routines.
"""

import json
import os
from typing import List, Dict, Any, Optional


class ExerciseRecommender:
    def __init__(self, exercise_db_path: Optional[str] = None):
        if exercise_db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            exercise_db_path = os.path.join(base_dir, 'data', 'exercises', 'exercises.json')
        
        self.exercise_db_path = exercise_db_path
        self.exercises = self._load_exercises()

    def _load_exercises(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.exercise_db_path):
            return []
        with open(self.exercise_db_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def recommend(self, state: str, severity_score: float, vitals: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Recommend personalized routines based on predicted state:
        - state: 'NORMAL', 'MILD ANXIETY', 'HIGH STRESS'
        - severity_score: 0 - 100 (where 0 is completely calm, 100 is severe acute crisis)
        """
        results = []
        for ex in self.exercises:
            target = ex.get('target_state', 'NORMAL')
            match_score = 0.0
            rationale = ""

            if state == "HIGH STRESS":
                if target == "HIGH STRESS":
                    match_score = 95.0 - (ex.get('duration_min', 15) * 0.5)
                    rationale = (
                        f"Chỉ số Cortisol và căng thẳng thần kinh giao cảm (SNS) đang ở ngưỡng cao. "
                        f"{ex['name']} được chỉ định lâm sàng để kích hoạt thần kinh đối giao cảm (PNS), "
                        f"hạ nhịp tim nghỉ và giảm co cứng cơ trục HPA."
                    )
                elif target == "MILD ANXIETY":
                    match_score = 65.0
                    rationale = "Có thể thực hiện sau khi đã hoàn thành các bài tập thở phục hồi."
                else:
                    match_score = 15.0
                    rationale = "CẢNH BÁO: Không khuyến nghị tập cường độ cao khi cơ thể đang chịu stress nặng."

            elif state == "MILD ANXIETY":
                if target == "MILD ANXIETY":
                    match_score = 92.0
                    rationale = (
                        f"Phát hiện dấu hiệu bồn chồn và biến thiên nhịp tim dao động. "
                        f"{ex['name']} giúp giải phóng năng lượng ứ đọng, tăng sản sinh BDNF và Dopamine mà không gây kiệt sức."
                    )
                elif target == "HIGH STRESS":
                    match_score = 80.0
                    rationale = "Hỗ trợ xoa dịu tâm trí nếu cảm thấy căng thẳng tăng dần."
                else:
                    match_score = 50.0
                    rationale = "Có thể tập nhẹ, tránh gắng sức tối đa."

            else:  # NORMAL
                if target == "NORMAL":
                    match_score = 96.0
                    rationale = (
                        f"Thể trạng sinh lý và độ biến thiên HRV rất lý tưởng! "
                        f"{ex['name']} giúp tối ưu hóa sức bền tim mạch, xây dựng mật độ cơ và nâng cao ngưỡng chịu tải của cơ thể."
                    )
                elif target == "MILD ANXIETY":
                    match_score = 78.0
                    rationale = "Bài tập duy trì độ dẻo dai và tính linh hoạt khớp cơ."
                else:
                    match_score = 60.0
                    rationale = "Dùng cho ngày phục hồi tích cực (Active Recovery)."

            item = dict(ex)
            item['match_score'] = round(float(match_score), 1)
            item['rationale'] = rationale
            results.append(item)

        # Sort descending by match_score
        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results
