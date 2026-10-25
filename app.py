import streamlit as st
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai

# ---------------------------------------------------------
# 1. 페이지 기본 설정 & 디자인
# ---------------------------------------------------------
st.set_page_config(page_title="AI 사진 크리틱 스튜디오", page_icon="📸", layout="wide")

st.title("📸 AI 프로페셔널 사진 크리틱 스튜디오")
st.caption("컴퓨터 비전(CV)의 정량적 구도 분석과 AI의 감성적 크리틱을 결합한 사진 평가 시스템입니다.")

# 사이드바 API 키 입력 및 설정
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("Gemini API Key를 입력하세요", type="password")
    st.markdown("---")
    st.markdown("""
    **분석 항목:**
    - 📐 수평선/수직선 기울기 (도)
    - 💡 섀도우 & 하이라이트 노출 상태
    - 🎨 프레임 구도 및 톤 앤 매너
    - 📝 프로 전문가 종합 크리틱
    """)

# ---------------------------------------------------------
# 2. CV 알고리즘 (수평, 노출, 구도 정량 분석)
# ---------------------------------------------------------
def analyze_photo_cv(image_bytes):
    # 이미지 디코딩
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # (1) 수평 기울기 검출 (Hough Line Transform)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=width * 0.25, maxLineGap=15)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -35 < angle < 35 and abs(angle) > 0.1:  # 수평 근처 주요 선 추출
                angles.append(angle)
    
    avg_tilt = np.mean(angles) if len(angles) > 0 else 0.0

    # (2) 히스토그램 노출 분석
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = height * width
    shadow_ratio = (np.sum(hist[:25]) / total_pixels) * 100     # 암부 뭉침 비율 (%)
    highlight_ratio = (np.sum(hist[230:]) / total_pixels) * 100 # 명부 손실 비율 (%)

    return {
        "tilt_angle": round(float(avg_tilt), 2),
        "shadow_ratio": round(float(shadow_ratio), 1),
        "highlight_ratio": round(float(highlight_ratio), 1),
        "width": width,
        "height": height
    }

# ---------------------------------------------------------
# 3. 메인 웹 인터페이스 및 실행 로직
# ---------------------------------------------------------
uploaded_file = st.file_uploader("크리틱을 받을 사진을 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 사진", use_container_width=True)

    if st.button("🚀 전문가 크리틱 시작하기", type="primary"):
        if not api_key:
            st.error("🔑 사이드바에 Gemini API Key를 입력해 주세요.")
        else:
            with st.spinner("사진의 구도, 노출, 감성을 정밀 분석 중입니다..."):
                # 1단계: OpenCV 데이터 계산
                image_bytes = uploaded_file.getvalue()
                cv_data = analyze_photo_cv(image_bytes)

                # 2단계: Gemini API 호출
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')

                prompt = f"""
                너는 20년 경력의 상업/예술 사진 작가이자 프로 사진 크리틱 전문가야.
                제시된 사진과 아래의 물리적 정량 분석 데이터를 종합해서 전문적이고 건설적인 사진 크리틱을 작성해줘.

                [사진 물리 데이터]
                - 수평선 측정 기울기: {cv_data['tilt_angle']}도
                - 섀도우(어두운 영역) 비율: {cv_data['shadow_ratio']}%
                - 하이라이트(밝은 영역) 비율: {cv_data['highlight_ratio']}%
                - 해상도: {cv_data['width']}x{cv_data['height']}

                [작성 가이드라인]
                1. 톤앤매너: 전문적이면서도 따뜻하고 명확한 피드백.
                2. 구체적 분석: 구도(삼분법, 수평, 프레임), 빛과 색감(명암 대비, 섀도우), 서사/스토리텔링(시선 유도, 피사체 간 상호작용).
                3. 반드시 아래 양식(Markdown)으로 출력할 것:

                ### 1. 사진의 명장면 (장점)
                - (3가지 항목으로 나누어 정밀 분석)

                ### 2. 아쉬운 점 & 보완할 부분
                - (구도, 노출, 시선 분산 요소 등 2~3가지)

                ### 3. 촬영 및 후보정 솔루션 (Action Plan)
                - **촬영 팁:** (앵글, 구도, 시선 관련)
                - **보정 팁:** (크롭 비율, 섀도우/하이라이트 조절, 수평 수직 정밀 교정 등)
                """

                try:
                    response = model.generate_content([prompt, image])
                    
                    st.success("분석이 완료되었습니다!")
                    st.markdown("---")

                    # CV 분석 지표 요약 표시
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📐 수평 기울기", f"{cv_data['tilt_angle']}°")
                    col2.metric("🌑 섀도우 뭉침", f"{cv_data['shadow_ratio']}%")
                    col3.metric("☀️ 하이라이트 손실", f"{cv_data['highlight_ratio']}%")

                    st.markdown("---")
                    # LLM 크리틱 리포트 출력
                    st.markdown(response.text)

                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {e}")
