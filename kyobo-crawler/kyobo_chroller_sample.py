import time
from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from playwright.sync_api import sync_playwright

# Playwright 초기화 (Selenium의 driver 역할)
# 스크립트 실행 시 브라우저가 열립니다.
# Playwright 초기화 (Selenium의 driver 역할)
p = sync_playwright().start()
browser = p.chromium.launch(headless=False)
page = browser.new_page()

url = "https://product.kyobobook.co.kr/detail/S000210621680"
page.goto(url)

# 리뷰 탭 클릭 (리뷰 목록 로드)
try:
    # 탭이 로드될 때까지 대기 (텍스트로 찾기)
    page.wait_for_selector('.tab_list_wrap', state='visible', timeout=10000)
    # '리뷰' 텍스트를 포함하는 탭 클릭
    page.locator("text=리뷰").first.click()
    page.wait_for_timeout(2000) # 탭 전환 대기
except Exception as e:
    print(f"리뷰 탭 이동 중 오류: {e}")

# 데이터 프레임 생성 (도서 리뷰용 컬럼으로 변경)
df_review_book = pd.DataFrame(columns=['date', 'rate', 'id', 'content'])

# 상품 리뷰 크롤링 함수
def review_crawling(df, target_page_count):    
    for current_page in range(1, target_page_count + 1):      
        print(f"Processing page {current_page}...")
        
        # 리뷰 아이템 로드 대기
        try:
            page.wait_for_selector('.comment_list', state='visible', timeout=5000)
        except:
            print("리뷰 리스트를 찾을 수 없습니다.")
            break

        # 현재 페이지의 모든 리뷰 아이템 가져오기
        reviews = page.locator('.comment_item')
        count = reviews.count()
        
        print(f"Found {count} reviews on page {current_page}")
        
        for i in range(count):
            try:
                review = reviews.nth(i)
                
                # 데이터 추출
                date = review.locator('.date').inner_text() if review.locator('.date').count() > 0 else ""
                # 평점은 '4점' 등의 텍스트로 되어 있거나 별점 이미지일 수 있음. 여기서는 텍스트 추출 시도.
                # klover_score 내의 텍스트 추출
                rate = review.locator('.klover_score').inner_text() if review.locator('.klover_score').count() > 0 else ""
                id_text = review.locator('.user_id').inner_text() if review.locator('.user_id').count() > 0 else ""
                content = review.locator('.comment_text').inner_text() if review.locator('.comment_text').count() > 0 else ""
                
                # DataFrame에 추가
                df.loc[len(df)] = [date, rate, id_text, content]

            except Exception as e:
                print(f"Error parsing review {i}: {e}")
                pass

        # 페이지네이션 처리
        try:
            # 다음 버튼이 있는지 확인
            next_btn = page.locator('.btn_page.next')
            if next_btn.count() > 0 and next_btn.is_visible():
                next_btn.click()
                page.wait_for_timeout(2000) # 페이지 로드 대기
            else:
                print("마지막 페이지입니다.")
                break
        except Exception as e:
            print(f"Pagination error: {e}")
            break
        
        print(f'{current_page}페이지 크롤링 완료')

# 스크립트 실행 부분
if __name__ == "__main__":
    print("교보문고 리뷰 크롤링 시작...")
    
    # 5페이지만 크롤링 시도 (필요에 따라 수정)
    review_crawling(df_review_book, 5)
    
    print("크롤링 완료")
    print(df_review_book.head())
    
    # 결과 저장
    df_review_book.to_csv('kyobo_reviews.csv', index=False, encoding='utf-8-sig')
    print("kyobo_reviews.csv 저장 완료")
    
    browser.close()
    p.stop()