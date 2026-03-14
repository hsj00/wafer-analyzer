# =============================================================================
# folder_picker_helper.py
# Streamlit 백그라운드 스레드 우회용 독립 프로세스
# =============================================================================
#
# [왜 이 파일이 필요한가]
# macOS에서 tkinter GUI는 반드시 프로세스의 '메인 스레드'에서 실행되어야 함.
# Streamlit은 스크립트를 백그라운드 스레드에서 실행하므로,
# 거기서 tk.Tk()를 호출하면 NSWindow 생성 실패 → SIGABRT 크래시.
#
# 해결 방법:
#   이 파일을 subprocess.run()으로 '별도 프로세스'로 실행
#   → 새 프로세스의 메인 스레드에서 tkinter 실행 → 크래시 없음
#   → 선택된 경로를 stdout으로 출력 → Streamlit이 결과를 수신
#
# [최적화 내역]
#   1. os, sys를 파일 최상단에 import (함수 내부 → 전역으로 이동)
#      이유: 함수 내부 import는 호출마다 sys.modules 탐색 오버헤드 발생.
#            최상단 import는 모듈 로딩 시 1회만 실행되어 더 효율적.
#   2. 초기 경로 결정 로직을 _get_initial_dir()로 분리
#      이유: 단일 책임 원칙 + 단독 테스트 가능
#   3. 경로 유효성 검증 추가 (isdir로 실제 존재 여부 확인)
#      이유: 존재하지 않는 경로를 initialdir로 넘기면 tkinter가 오류를 내거나
#            예측 불가한 동작을 할 수 있음
# =============================================================================

# ★ 최적화 [1]: 모든 import를 파일 최상단에 배치
import os  # 경로 처리: os.path.expanduser, os.path.isdir
import sys  # 커맨드라인 인자(sys.argv), 강제 종료(sys.exit)

# ★ tkinter 미설치 시 즉시 종료
# 이유: ImportError를 방치하면 알 수 없는 크래시처럼 보여 디버깅이 어려움.
#       아무 출력 없이 종료하면 Streamlit에서 stdout == "" → None 반환
#       → 인라인 브라우저로 자동 폴백하여 UX가 유지됨.
try:
    import tkinter as tk  # GUI 툴킷 (macOS/Windows 기본 내장)
    from tkinter import filedialog  # 폴더 선택 전용 다이얼로그 모듈
except ImportError:
    sys.exit(0)  # 아무것도 출력하지 않고 종료 → Streamlit이 None으로 처리


def _get_initial_dir() -> str:
    """
    ★ 최적화 [2]: 초기 경로 결정 로직을 별도 함수로 분리.

    우선순위:
      1순위: sys.argv[1]이 존재하고 실제 폴더이면 사용
      2순위: 홈 폴더(~)로 폴백

    [sys.argv 구조]
      sys.argv[0] = 이 스크립트 파일명 (항상 존재)
      sys.argv[1] = Streamlit이 전달한 초기 폴더 경로 (선택적)
                    wafer_app_global.py에서 subprocess.run() 호출 시 전달됨
    """
    if len(sys.argv) > 1:
        candidate = sys.argv[1]
        # ★ 최적화 [3]: os.path.isdir로 실제 존재 여부 검증
        # 존재하지 않는 경로를 tkinter에 넘기면 동작이 OS마다 달라짐
        if os.path.isdir(candidate):
            return candidate

    # os.path.expanduser("~"):
    #   macOS/Linux → /Users/username 또는 /home/username
    #   Windows     → C:\Users\username
    # 루트("/")보다 홈 폴더가 사용자에게 더 실용적인 시작점
    return os.path.expanduser("~")


def main():
    """
    tkinter 폴더 선택 다이얼로그를 실행하고 결과를 stdout에 출력.

    실행 흐름:
      1. 초기 디렉터리 결정 (_get_initial_dir)
      2. tkinter 루트 창 생성 후 즉시 숨김 (깜빡임 방지)
      3. 창을 최상위로 고정 (다른 앱 뒤에 숨는 현상 방지)
      4. 폴더 선택 다이얼로그 표시
      5. tkinter 리소스 해제 (메모리 누수 방지)
      6. 선택 결과 출력 (취소 시 무출력)
    """
    initial = _get_initial_dir()

    # tkinter 루트 창 생성 후 withdraw()로 즉시 숨김.
    # withdraw() 없이 진행하면 빈 회색 창이 잠깐 화면에 깜빡이며 나타나
    # 사용자에게 불필요한 창이 보임.
    root = tk.Tk()
    root.withdraw()

    # '-topmost' True: 이 창을 화면의 모든 창 위에 표시.
    # 이 설정 없이는 macOS에서 Streamlit 브라우저 뒤에 숨어버려
    # 사용자가 "폴더 선택 창이 열리지 않는다"고 착각하는 상황 발생.
    root.call('wm', 'attributes', '.', '-topmost', True)

    # filedialog.askdirectory: 파일이 아닌 '폴더'만 선택 가능한 전용 다이얼로그.
    # 사용자가 Cancel 버튼을 누르면 빈 문자열("") 반환.
    folder = filedialog.askdirectory(
        master=root,
        title="웨이퍼 데이터 폴더 선택",
        initialdir=initial
    )

    # root.destroy(): 루트 창과 모든 하위 tkinter 위젯을 메모리에서 해제.
    # 이를 생략하면 일부 OS에서 프로세스가 완전히 종료되지 않아
    # 좀비 프로세스 또는 메모리 누수가 발생할 수 있음.
    root.destroy()

    # 폴더를 선택한 경우에만 경로를 stdout으로 출력.
    #
    # end="" → 줄바꿈(\n) 없이 출력.
    #   Streamlit에서 result.stdout.strip()으로 수신하므로 \n 불필요.
    #   \n이 포함되면 os.path.isdir() 검증 실패 가능성 있음.
    #
    # 취소 시 (folder == ""):
    #   아무것도 출력하지 않음 → Streamlit: stdout == "" → None 반환
    #   → show_folder_browser = True → 인라인 브라우저로 자동 폴백
    if folder:
        print(folder, end="")


# Python 실행 방식에 따른 진입점 제어:
#   직접 실행: python folder_picker_helper.py  → __name__ == "__main__" → main() 호출
#   모듈 import: import folder_picker_helper   → __name__ == 모듈명    → main() 미호출
#
# subprocess.run([sys.executable, helper, initial])은 직접 실행이므로 항상 main() 호출됨.
if __name__ == "__main__":
    main()