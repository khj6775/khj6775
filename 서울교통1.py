import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertForMaskedLM

model = BertForMaskedLM.from_pretrained("remi/bertabs-finetuned-extractive-abstractive-summarization")


class ChronosT5Model:
    def __init__(self, model_name='amazon/chronos-t5-tiny'):
        # Chronos-T5 모델과 토크나이저 불러오기
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def preprocess_data(self, speed_data_path):
        # 데이터 불러오기
        self.speed_data = pd.read_csv(speed_data_path, encoding='utf-8-sig')
        
        # 필요한 컬럼 선택 (시간대, 차량속도 등)
        time_columns = [f'{str(i).zfill(2)}시' for i in range(1, 25)]
        required_columns = ['일자', '요일', '도로명', '지점명', '방향'] + time_columns
        self.speed_data = self.speed_data[required_columns]
        
        # 시계열 데이터를 모델 입력 형식으로 변환
        inputs = []
        for idx, row in self.speed_data.iterrows():
            input_str = f"date: {row['일자']} road: {row['도로명']} speed: {row[time_columns].values}"
            inputs.append(input_str)
        
        return inputs

    def predict_congestion(self, inputs):
        predictions = []
        for input_str in inputs:
            # 입력을 토큰화
            input_ids = self.tokenizer(input_str, return_tensors="pt").input_ids
            
            # 모델 예측
            outputs = self.model.generate(input_ids)
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(prediction)
        
        return predictions

def main():
    # 모델 초기화 (Hugging Face에서 바로 불러오기)
    model_name = 'amazon/chronos-t5-tiny'
    chronos_model = ChronosT5Model(model_name=model_name)

    # CSV 파일 경로
    speed_data_path = "C:/ai5/_data/AiDA/data/잘못뽑은데이터csv/필터링된_통합_차량통행속도_데이터.csv"
    
    # 데이터 전처리
    inputs = chronos_model.preprocess_data(speed_data_path)
    
    # 예측 수행
    predictions = chronos_model.predict_congestion(inputs)
    
    # 예측 결과 출력
    for i, prediction in enumerate(predictions[:5]):  # 일부 결과만 출력
        print(f"Prediction {i+1}: {prediction}")

if __name__ == "__main__":
    main()