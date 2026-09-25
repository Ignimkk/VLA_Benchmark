from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MPath

OUT = Path(__file__).resolve().parent
for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']:
    fm.fontManager.addfont(f)
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'svg.fonttype':'path','pdf.fonttype':42})
COL={'blue':('#eaf1f7','#436d91'),'orange':('#fcf1e3','#9e6a2c'),'red':('#f7eaea','#9e5454'),'purple':('#f0edf7','#75628e'),'green':('#fbfdfb','#48745a'),'gray':('#f6f7f8','#545d65')}
TEXT='#17212a'
def canvas(w,h):
    fig,ax=plt.subplots(figsize=(w/10,h/10));fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,w),ylim=(0,h));ax.axis('off');return fig,ax

def txt(ax,x,y,s,size=12,weight='normal',ha='center',va='center',color=TEXT,linespacing=1.5):
    return ax.text(x,y,s,fontsize=size,weight=weight,ha=ha,va=va,color=color,linespacing=linespacing,zorder=5)

def box(ax,x,y,w,h,title,body='',col='gray',fs=13,bs=11,lw=1.3):
    fill,edge=COL[col]
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.7',facecolor=fill,edgecolor=edge,linewidth=lw,zorder=2))
    if body:
        txt(ax,x+w/2,y+h*.72,title,fs,'bold')
        txt(ax,x+w/2,y+h*.31,body,bs,linespacing=1.45)
    else: txt(ax,x+w/2,y+h/2,title,fs,'bold')

def arr(ax,points,label=None,lp=None,dash=False,color='#47515a',lw=1.4,size=12):
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=size,linewidth=lw,linestyle=(0,(4,3)) if dash else '-',color=color,zorder=3))
    if label: txt(ax,*lp,label,10)

def save(fig,name):
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=180,facecolor='white')
    plt.close(fig)


def label(ax,x,y,s,fs=10):
    t=txt(ax,x,y,s,fs)
    t.set_bbox(dict(facecolor='white',edgecolor='none',pad=1.5))
    return t

def future(ax,x,y,w,h):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.7',fill=False,edgecolor='#8a660d',linewidth=2,zorder=4))

# 01: data ownership and execution responsibility.
fig,ax=canvas(160,90)
txt(ax,80,85.5,'VLA + AG3S + trajopt',22,'bold')
txt(ax,80,80,'목표 구조 · attention은 조작 대상을, 기하는 충돌 가능성을 정한다',12)
box(ax,15,66,44,10,'카메라 관측 · 로봇 상태','RGB / depth · 보정값 · 촬영 시점 자세',fs=12,bs=9)
box(ax,78,66,64,10,'VLA (π0.5)','언어 지시 + RGB → action chunk · attention',fs=14,bs=10)
arr(ax,[(59,71),(78,71)])
label(ax,68.5,73.8,'RGB',9)
box(ax,12,27,134,34,'',col='green',lw=2.8)
txt(ax,18,57.5,'AG3S',17,'bold',ha='left',color='#33573f')
box(ax,18,40,37,11,'환경 기하','로봇 제거 → cuRobo 거리장','blue',12,10)
box(ax,63,40,34,11,'동작 영역 · 계획','FK → swept-volume ROI','orange',12,9.5)
box(ax,105,40,35,11,'조작 대상','attention → grounding → 잠금*','red',12,9.2)
arr(ax,[(37,66),(37,51)])
label(ax,37,63.5,'depth · 로봇 상태',9)
arr(ax,[(86,66),(86,51)])
label(ax,86,63.5,'action chunk',9)
arr(ax,[(122.5,66),(122.5,51)])
label(ax,122.5,63.5,'attention',9)
arr(ax,[(63,45.5),(55,45.5)],color=COL['orange'][1])
label(ax,59,48.2,'ROI',9)
box(ax,18,29.5,122,7,'충돌 제약 · CollisionConstraintSet','거리장 + 조작 대상 + 구별 마진 + validity / phase','purple',12,10)
arr(ax,[(37,40),(37,36.5)])
arr(ax,[(122.5,40),(122.5,36.5)])
box(ax,42,15,76,8,'trajopt','SQP → QP → 전 행 재검사',fs=13,bs=10.5)
arr(ax,[(80,29.5),(80,23)])
arr(ax,[(142,19),(118,19)])
label(ax,131,22.5,'VLA 원본 chunk',9)
box(ax,38,2,84,7,'과제 소유자 · 실행 / 유지 / 재계획 결정','보정 chunk + 상태 + 위반량을 확인',fs=11.5,bs=9.5)
arr(ax,[(80,15),(80,9)])
arr(ax,[(122,5.5),(151,5.5),(151,45.5),(140,45.5)],dash=True)
label(ax,135,10,'외부 성공 판정 → detach',8.5)
txt(ax,15,24.2,'* 잠금·ROI·목적지 마진은 계획 요소',8.5,ha='left')
save(fig,'01_overall')

# 02: real multiview order; cuRobo reduced to a single interface box.
fig,ax=canvas(180,225)
txt(ax,90,218,'AG3S · 대상 식별에서 충돌 제약까지',22,'bold')
txt(ax,90,211,'로그와 코드에 근거한 목표 구조 · “계획” 표시는 미구현 또는 통합 전',11.5)
# Roles and three independent sources
for x,w,c,h in [(6,42,'orange','동작 영역'),(57,48,'blue','기하 관측'),(114,48,'red','대상 의미')]:
    txt(ax,x+w/2,205.5,h,13,'bold',color=COL[c][1])
box(ax,6,190,42,12,'VLA action chunk','동작 묶음 + 현재 로봇 자세','orange',12,10)
box(ax,57,190,48,12,'depth × 3 · 로봇 상태','카메라 보정값 · 촬영 시점 자세','blue',12,10)
box(ax,114,190,48,12,'VLA attention','카메라별 맵 · 실제 이미지 해상도','red',12,10)
# Pair of camera-local branches followed by a shared fusion.
ax.add_patch(Rectangle((54.5,125),110,62,facecolor='#f6f8fa',edgecolor='#b6bfc7',linewidth=.8,zorder=0))
txt(ax,109.5,185.7,'카메라마다 처리한 뒤 하나의 장면으로 통합',9)
box(ax,57,170,48,12,'1. 점구름 재구성','핀홀 역투영 · base 좌표 · uv 보존','blue',12,10)
box(ax,114,170,48,12,'attention 어댑터','선택한 layer / head → 픽셀맵 (H, W)','red',12,10)
box(ax,57,150,48,12,'2. 로봇 자기 몸 제외','점 필터 + depth 픽셀 마스크','blue',12,10)
box(ax,114,150,48,12,'3. lifting · 점에 attention 부착','uv로 대응 · 낮은 attention도 점은 유지','red',11.5,9.5)
for x in [81,138]:
    arr(ax,[(x,190),(x,182)])
    arr(ax,[(x,170),(x,162)])
arr(ax,[(105,156),(114,156)])
label(ax,109.5,165,'점 + uv',9)
box(ax,57,127,105,14,'4. 멀티뷰 융합 → 정규화','base 격자마다 관측점 유지 · 원시 attention을 max로 합친 뒤 정규화','purple',13,10.4)
for x in [81,138]:arr(ax,[(x,150),(x,141)])
# Grounding depends on both the fused cloud and support exclusion mask.
box(ax,57,104,48,13,'5. 지지면 분할 · RANSAC','grounding이 테이블로 번지지 않게 분리','blue',12,9.6)
box(ax,114,104,48,13,'6. grounding · 주목 대상','씨앗 → 3D 연결성 → 덩어리 → 점수','red',12,10)
for x in [81,138]:arr(ax,[(x,127),(x,117)])
arr(ax,[(105,110.5),(114,110.5)],color=COL['blue'][1])
label(ax,109.5,121,'지지면 마스크',8.6)
txt(ax,81,94,'지지면은 분할에 사용\n표면은 ESDF에 유지 · 평면 제약은 기본 OFF',9.3,linespacing=1.55)
# Action-conditioned region is explicitly planned, independent from attention.
box(ax,6,170,42,12,'순기구학(FK) → 로봇 구 궤적','chunk의 각 스텝에서 구 위치 계산','orange',11,9.4)
box(ax,6,145,42,14,'swept-volume ROI · 계획','로봇이 쓸고 지나갈 부피 + 여유 폭','orange',11.8,9.2)
future(ax,6,145,42,14)
arr(ax,[(27,190),(27,182)])
arr(ax,[(27,170),(27,159)])
arr(ax,[(44,145),(44,68),(57,68)],color=COL['orange'][1])
label(ax,49.5,64.8,'ROI',9)
txt(ax,8,132,'정밀 영역의 기준은\nattention의 주목 대상이 아니라\n로봇이 지나갈 공간이다.',10,ha='left',va='top',linespacing=1.65)
# Only masked depth reaches the field; point cloud/plane fitting is a separate path.
arr(ax,[(57,156),(52,156),(52,76),(57,76)],color=COL['blue'][1])
label(ax,52,86.5,'depth +\n로봇 마스크',8.5)
box(ax,57,65,48,14,'cuRobo · 거리장(ESDF)','거리 d(p) · 기울기 ∇d(p) 제공','blue',13,10.5)
# Episode-level identity and externally confirmed held geometry are different.
box(ax,114,83,48,15,'7. 조작 대상 잠금 · 계획','걸기: N프레임 일치 + 점수 격차\n유지: attention 이동 무시 · 해제: 외부 detach','red',12,9.1)
future(ax,114,83,48,15)
arr(ax,[(138,104),(138,98)])
box(ax,114,62,48,15,'8. 쥔 물체 표현 · attached','외부 attach로 등록 · parent_link 추종\n로봇 구 편입의 TO 연결은 남은 작업','red',11.8,9.3)
arr(ax,[(138,83),(138,77)])
txt(ax,171,108,'외부\n과제 소유자',8.5,linespacing=1.5)
arr(ax,[(171,103),(171,90.5),(162,90.5)],dash=True)
txt(ax,171,79.5,'성공 판정\n→ 잠금 해제\n(detach)',8,linespacing=1.4)
# Policy encodes permissions; it never deletes object geometry.
arr(ax,[(81,65),(81,52)])
arr(ax,[(138,62),(138,52)])
box(ax,6,25,156,27,'',col='purple',lw=1.8)
txt(ax,84,47,'접촉 권한 · (로봇 구, 대상) 쌍의 마진',14,'bold')
xs=[9,53,88,111,134];ws=[44,35,23,23,25]
heads=['조작 대상 · 권한 링크','목적지 · 계획','장애물','지지면','미지 조합']
vals=['phase별 완화 → 파지 시 0','별도 얇은 마진','full','full','full']
for x,w,h,v in zip(xs,ws,heads,vals):
    ax.add_patch(Rectangle((x,32),w,10,facecolor='white',edgecolor='#8a660d' if x==53 else '#aaa1b6',linewidth=2 if x==53 else .8,zorder=3))
    txt(ax,x+w/2,39,h,10,'bold');txt(ax,x+w/2,34.6,v,9.3)
txt(ax,84,28.5,'0도 관통 금지 · 비권한 링크는 full · 미지 조합은 full margin (fail-closed)',10)
box(ax,25,7,118,13,'CollisionConstraintSet → trajopt','거리장 · 조작 대상 · 구별 마진 · phase / 활성 손 · validity / status','purple',14,10)
arr(ax,[(84,25),(84,20)])
txt(ax,90,2.5,'phase·활성 손·파지/해제는 외부 입력  |  AG3S는 판정 정보를 제공하고, 실행 결정은 과제 소유자가 한다.',9)
save(fig,'02_ag3s_detail')
print(OUT)
