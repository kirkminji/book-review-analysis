#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
교보문고 리뷰 크롤러
- Windows와 Ubuntu 모두에서 작동
- UV로 실행 가능
"""

import requests
import pandas as pd
import json
from typing import List, Dict
import time
import sys
import platform

# Windows 콘솔 인코딩 설정 (Windows에서만)
if platform.system() == 'Windows':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

def fetch_book_reviews(sale_cmdtid: str, max_pages: int = 5) -> List[Dict]:
    """
    교보문고 API에서 책 리뷰 정보를 가져옵니다.
    
    Args:
        sale_cmdtid: 상품 ID (예: S000217467412)
        max_pages: 가져올 최대 페이지 수
    
    Returns:
        리뷰 정보 리스트
    """
    base_url = "https://product.kyobobook.co.kr/api/review/list"
    all_reviews = []
    
    # 헤더 설정
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
    }
    
    for page in range(1, max_pages + 1):
        params = {
            'page': page,
            'pageLimit': 10,
            'reviewSort': '001',
            'revwPatrCode': '002',
            'saleCmdtids': sale_cmdtid,
            'webToonYsno': 'N',
            'allYsno': 'N',
            'revwSummeryYn': 'Y',
            'saleCmdtid': sale_cmdtid
        }
        
        try:
            print(f"[페이지 {page}] 데이터 수집 중...")
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # 리뷰 데이터 추출
            if 'data' in data and 'reviewList' in data['data']:
                reviews = data['data']['reviewList']
                total_count = data['data'].get('totalCount', 0)
                
                if not reviews:
                    print(f"  >> 페이지 {page}에서 더 이상 리뷰가 없습니다.")
                    break
                
                all_reviews.extend(reviews)
                print(f"  >> 페이지 {page}: {len(reviews)}개 수집 (전체: {total_count}개)")
            else:
                print("  >> 리뷰 데이터를 찾을 수 없습니다.")
                break
            
            # API 요청 간격 (서버 부하 방지)
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            print(f"  >> 페이지 {page} 수집 중 오류 발생: {e}")
            break
    
    return all_reviews

def parse_reviews_to_dataframe(reviews: List[Dict]) -> pd.DataFrame:
    """
    리뷰 데이터를 DataFrame으로 변환합니다.
    """
    parsed_data = []
    
    for review in reviews:
        row = {
            '제목': review.get('cmdtName', '정보 없음'),
            '상품코드': review.get('cmdtcode', ''),
            '상품ID': review.get('saleCmdtid', ''),
            '리뷰_내용': review.get('revwCntt', '').strip(),
            '평점': review.get('revwRvgr', 0),
            '감정키워드': review.get('revwEmtnKywrName', ''),
            '작성자': review.get('mmbrId', '익명'),
            '작성일': review.get('cretDttm', '')[:10],  # 날짜만 추출
            '도움됨': review.get('reviewRecommendCount', 0),
            '댓글수': review.get('reviewCommentCount', 0)
        }
        parsed_data.append(row)
    
    return pd.DataFrame(parsed_data)

def main():
    """
    메인 실행 함수
    """
    # 크롤링할 상품 ID (기본값)
    # sale_cmdtid = "S000217467412"
    sale_cmdtid = "S000000610612"
    max_pages = 30  # 최대 30페이지 = 300개 리뷰
    
    # 명령줄 인자 처리
    if len(sys.argv) > 1:
        sale_cmdtid = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            max_pages = int(sys.argv[2])
        except ValueError:
            print("경고: 잘못된 페이지 수 입력, 기본값 30 사용")
    
    print("=" * 70)
    print("[교보문고 리뷰 크롤링 시작]")
    print("=" * 70)
    print(f"상품 ID: {sale_cmdtid}")
    print(f"최대 페이지: {max_pages}")
    print()
    
    # 리뷰 데이터 수집
    reviews = fetch_book_reviews(sale_cmdtid, max_pages=max_pages)
    
    if not reviews:
        print("\n[오류] 수집된 리뷰가 없습니다.")
        return
    
    print(f"\n[완료] 총 {len(reviews)}개의 리뷰를 수집했습니다.")
    print()
    
    # DataFrame 생성
    df = parse_reviews_to_dataframe(reviews)
    
    # CSV 파일로 저장
    output_filename = f'kyobo_reviews_{sale_cmdtid}.csv'
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print("=" * 70)
    print(f"[저장 완료] '{output_filename}'")
    print("=" * 70)
    print()
    
    # 데이터 미리보기
    print("[데이터 미리보기 - 처음 3개 리뷰]")
    print("-" * 70)
    for idx, row in df.head(3).iterrows():
        print(f"\n[리뷰 {idx+1}]")
        print(f"  제목: {row['제목']}")
        print(f"  평점: {row['평점']}점 / 감정: {row['감정키워드']}")
        print(f"  작성자: {row['작성자']} / 작성일: {row['작성일']}")
        content = row['리뷰_내용'][:80].replace('\n', ' ')
        print(f"  내용: {content}...")
    
    print()
    print("=" * 70)
    print("[통계 정보]")
    print("-" * 70)
    print(f"  총 리뷰 수: {len(df):,}개")
    print(f"  평균 평점: {df['평점'].mean():.2f}점")
    print(f"  최고 평점: {df['평점'].max()}점")
    print(f"  최저 평점: {df['평점'].min()}점")
    print()
    print("  평점 분포:")
    for score in sorted(df['평점'].unique(), reverse=True):
        count = len(df[df['평점'] == score])
        percentage = (count / len(df)) * 100
        bar = '█' * int(percentage / 2)
        print(f"    {score}점: {count:3}개 ({percentage:5.1f}%) {bar}")
    
    print()
    print("  감정 키워드 분포:")
    emotion_counts = df['감정키워드'].value_counts()
    for emotion, count in emotion_counts.head(5).items():
        percentage = (count / len(df)) * 100
        print(f"    {emotion}: {count}개 ({percentage:.1f}%)")
    
    print("=" * 70)
    print()
    print(f"💾 CSV 파일이 저장되었습니다: {output_filename}")
    print("📊 엑셀에서 열면 한글이 정상적으로 표시됩니다.")

if __name__ == "__main__":
    main()
