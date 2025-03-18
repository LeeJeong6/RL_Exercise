import time
import tracemalloc

def measure_performance(func):
    def wrapper(*args, **kwargs):
        # 시작 시간 측정
        start_time = time.time()
        
        # 메모리 측정을 시작
        tracemalloc.start()
        
        # 함수 실행
        result = func(*args, **kwargs)
        
        # 함수 실행 후 시간 측정
        end_time = time.time()
        
        # 메모리 측정 종료
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 실행 시간과 메모리 사용량 출력
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        print(f"Memory used: {current / 1024:.2f} KB (current), {peak / 1024:.2f} KB (peak)")
        
        return result
    
    return wrapper