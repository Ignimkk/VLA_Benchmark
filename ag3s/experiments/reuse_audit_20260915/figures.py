"""Render the audit evidence and proposed architecture; all inputs are saved experiment results."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
OUT=Path('benchmark/ag3s/docs/figures/reuse-audit-20260915')
def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=160,bbox_inches='tight')
 plt.close(fig)
summary={}
for name in ['rollout4','rollout5','rollout4_all','rollout5_all']:
 d=json.loads((OUT/f'{name}.json').read_text());fs=d['frames']
 before=np.array([f['clearance_before_mm'] for f in fs]);after=np.array([f['clearance_after_mm'] for f in fs])
 summary[name]={'n':len(fs),'cameras':d['cameras'],'before_negative':int((before<0).sum()),'fixed_geometric':int(((before<0)&(after>=0)).sum()),'improved':int((after>before).sum()),'status':dict(Counter(f['status'] for f in fs)),'after_min_mm':float(after.min()),'after_median_mm':float(np.median(after)),'reference_deviation_median':float(np.median([f['reference_deviation'] for f in fs])),'unknown_median':float(np.median([f['esdf_unknown'] for f in fs]))}
(OUT/'rollout_summary.json').write_text(json.dumps(summary,indent=2))
fig,axs=plt.subplots(2,2,figsize=(14,8),layout='constrained')
for col,n in enumerate([4,5]):
 fs=json.loads((OUT/f'rollout{n}_all.json').read_text())['frames'];x=[f['i'] for f in fs]
 for key,label,c in [('clearance_before_mm','보정 전','#aa4040'),('clearance_after_mm','보정 후','#21836b')]:axs[0,col].plot(x,[f[key] for f in fs],'o-',ms=3,label=label,color=c)
 axs[0,col].axhline(0,color='gray',lw=1);axs[0,col].set(title=f'run_000{n} · 3대 카메라 · 22 청크',ylabel='모델 기준 최소 여유거리 (mm)');axs[0,col].legend()
 axs[1,col].plot(x,[f['reference_deviation'] for f in fs],'o-',color='#385da0',ms=3)
 axs[1,col].set(xlabel='기록 청크 번호',ylabel='기준 궤적과의 전체 L2 차이')
 for ax in axs[:,col]:
  ax.axvspan(15.5,18.5,color='#e6ca86',alpha=.25);ax.grid(alpha=.2)
fig.suptitle('오프라인 재생: 보정 청크를 로봇에 실행하지 않음\n부착·목적지 주입 비활성 / 음영은 이전 로그의 담기 관찰 구간, 검출된 phase 아님',fontsize=15)
save(fig,'recorded-results')
fig,axs=plt.subplots(2,2,figsize=(14,8),layout='constrained')
ax=axs[0,0];ax.axhline(0,color='gray');ax.scatter([-30,0,45],[0,0,0],s=[150,80,150],c=['#1a997a','#111111','#c74e49']);ax.set(xlim=(-65,85),ylim=(-1,1),yticks=[],xlabel='합성 씬 x (mm)',title='다른 거리에서도 큰 마진 장애물을 놓침')
for x,t in [(-30,'목적지\n거리 30 / 마진 20'),(0,'질의점'),(45,'장애물\n거리 45 / 마진 50')]:ax.text(x,.15,t,ha='center',fontsize=10)
ax.text(0,-.65,'현재 +10 mm  →  모든 쌍 기준 −5 mm',ha='center',color='#a62e2e',weight='bold')
axs[0,1].bar(['현재 최근접 라벨','모든 대상의 최솟값'],[10,-5],color=['#21836b','#c74e49']);axs[0,1].axhline(0,color='gray');axs[0,1].set(ylabel='여유거리 (mm)',title='Production 소비 함수에서 재현')
axs[1,0].plot([9.9,10.1],[14.95,-14.95],'o-',color='#b74b4b');axs[1,0].set(xlabel='합성 보간 격자 x (mm)',ylabel='여유거리 (mm)',title='라벨 경계: 0.2 mm 이동에 29.9 mm 점프')
axs[1,1].plot([19.9,20.1],[10,40],'o-',color='#385da0');axs[1,1].set(xlabel='합성 미세 창 경계 x (mm)',ylabel='합성 거리 (mm)',title='계층 경계: 0.2 mm 이동에 30 mm 점프')
fig.suptitle('충돌 정책·거리장 반례 / 각 패널은 독립된 합성 조건',fontsize=16);save(fig,'counterexamples')
r=np.load(OUT/'gpu_reprojection.npz');p=r['centers'];sel=r['visible'];fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for ax,key,title in zip(axs[:2],['stored_feature','current_attention'],['cuRobo 표면의 저장 특성','같은 표면에 현재 attention 재투영']):
 s=ax.scatter(p[sel,0],p[sel,1],c=r[key][sel],s=7,vmin=0,vmax=1,cmap='viridis');ax.set(title=title,xlabel='camera x (m)',ylabel='camera y (m)',aspect='equal')
fig.colorbar(s,ax=axs[:2],label='합성 attention')
axs[2].bar(['첫 관측','둘째 관측의 입력','둘째 적분 후'],[1,0,.5],color=['#385da0','#ccc','#d29539']);axs[2].set(ylim=(0,1.12),ylabel='attention 값',title='별도 시간 실험: 단순 누적의 지연');axs[2].tick_params(axis='x',labelsize=9)
fig.suptitle('GPU 실험: z=1 m 평면을 카메라 정면에서 봄 / 실제 물체 분할 성능은 미검증',fontsize=15);save(fig,'gpu-mapping')
# Concise architecture: ESDF deliberately summarized.
fig,ax=plt.subplots(figsize=(14,7.5));ax.set(xlim=(0,14),ylim=(0,7.5));ax.axis('off')
def box(x,y,w,h,t,c='#e9eef5'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc=c,ec='#526780',lw=1));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=11)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#526780',lw=1.6))
box(.3,5.6,3.2,1.0,'동기화된 RGB-D · 촬영 자세\n로봇 표면 + depth 일치 마스크')
box(4.3,5.6,3.4,1.0,'cuRobo: 공유 기하 지도\nTSDF · ESDF · 표면 추출','#dcebd9')
box(8.6,5.6,4.7,1.0,'현재 attention → 가시 표면에 재투영\n대상 후보 · 물체 ID · 신뢰도','#e5def3')
arrow((3.5,6.1),(4.3,6.1));arrow((7.7,6.1),(8.6,6.1))
box(8.6,3.4,4.7,1.15,'AG3S: 조작 대상 잠금 · 목적지 · phase\nattach / 유지 / detach\n파지 형상은 물체 좌표계로 보존','#e5def3');arrow((10.95,5.6),(10.95,4.55))
box(4.3,3.4,3.4,1.15,'정책 종류별 / 객체별 거리 질의\n쌍별 마진 적용 후 최솟값\n관측·범위·지도 시각 포함');arrow((6,5.6),(6,4.55));arrow((8.6,4.0),(7.7,4.0))
box(.3,3.4,3.2,1.15,'VLA 기준 청크\n관절 시퀀스 · 그리퍼 채널');arrow((1.9,3.4),(1.9,2.1))
box(.3,.95,7.4,1.15,'기존 SQP 우선 유지: 의도 보존 비용 + 충돌 제약\n후보 궤적 전체 · 파지 물체 · 비권한 링크 · 스텝 사이 검사','#dce9f7');arrow((6,3.4),(6,2.1))
box(8.6,.95,4.7,1.15,'원래 시간축·그리퍼를 유지한 보정 청크\n제약 위반 / 미관측 / 범위 이탈 시 실패 반환');arrow((7.7,1.5),(8.6,1.5))
ax.text(7,7.2,'권고 아키텍처 — 구현 전 검토안',ha='center',fontsize=19,weight='bold');ax.text(7,.3,'AG3S가 기하를 중복 적분할 필요는 없음. 물체 정체와 접촉 권한은 별도 책임.',ha='center',fontsize=12)
save(fig,'architecture-simple')
# Detailed companion is an editable Mermaid artifact (keeps contracts readable).
print(json.dumps(summary,indent=2))
fig,ax=plt.subplots(figsize=(16,11));ax.set(xlim=(0,16),ylim=(0,11));ax.axis('off')
# box/arrow helpers above deliberately bind the current ax.
def named_box(x,y,w,h,title,body,c='#e9eef5'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc=c,ec='#526780',lw=1))
 ax.text(x+w/2,y+h*.72,title,ha='center',va='center',fontsize=11.5,weight='bold')
 ax.text(x+w/2,y+h*.34,body,ha='center',va='center',fontsize=9.8,linespacing=1.35)
named_box(.3,8.8,4.5,1.25,'센서 입력',
          'RGB-D · intrinsics K · camera pose T(t)\n단위 · 동기화 · depth 품질 검사')
named_box(5.6,8.8,4.6,1.25,'로봇 마스크',
          '로봇 rendering depth와 관측 depth 비교\n로봇 픽셀만 제거 · 주변 물체 보존')
named_box(11,8.8,4.5,1.25,'cuRobo 환경 지도',
          '공유 TSDF · ESDF · 표면 추출','#dcebd9')
arrow((4.8,9.3),(5.6,9.3));arrow((10.2,9.3),(11,9.3))
named_box(.3,6.5,4.5,1.35,'VLA attention',
          '현재 점수와 persistent object ID 분리\n저장 feature는 후속 비교 후보','#e5def3')
named_box(5.6,6.5,4.6,1.35,'3D attention 투영',
          '추출 표면을 각 카메라에 재투영\n시야 · occlusion · depth 일치 검사\nraw 점수 융합 후 정규화','#e5def3')
named_box(11,6.5,4.5,1.35,'장면 상태',
          '정적 환경과 동적 물체 형상 분리\nmap version · 관측 여부 · last seen\nattach 시 해당 물체만 환경에서 제외')
arrow((2.55,8.8),(2.55,7.65));arrow((4.8,7.05),(5.6,7.05));arrow((13.25,8.8),(13.25,7.65));arrow((11,7.1),(10.2,7.1))
named_box(.3,4.15,4.5,1.35,'Target grounding',
          '물체 경계 · 후보 ID · confidence\n조작 대상 잠금 · 목적지 별도 입력\n모호하면 접촉 권한을 추가하지 않음','#e5def3')
named_box(5.6,4.15,4.6,1.35,'파지 상태',
          'attach → FK로 이동 → detach\n형상 범위 · 미관측 면 · slip 상태','#e5def3')
named_box(11,4.15,4.5,1.35,'접촉 정책',
          'link × object × phase\n정책 그룹별 거리 또는 object query\nmin_j(distance − radius − margin_ij)')
arrow((7.9,6.5),(7.9,5.95));arrow((7.9,5.95),(2.55,5.95));arrow((2.55,5.95),(2.55,5.3));arrow((4.8,4.7),(5.6,4.7));arrow((10.2,4.7),(11,4.7));arrow((13.25,6.5),(13.25,5.3))
named_box(.3,1.7,4.5,1.4,'궤적 보정',
          'VLA 기준 chunk를 기존 SQP로 보정\n기준 궤적 추종 + joint · velocity · collision 제약\ngripper channel과 timing 보존','#dce9f7')
named_box(5.6,1.7,4.6,1.4,'최종 안전 검증',
          '후보 궤적 전체를 다시 검사\n환경 + 파지 물체 ↔ 비허용 link\n관측 범위 · ROI · inter-step collision','#dce9f7')
named_box(11,1.7,4.5,1.4,'실행 판단',
          '보정 chunk + status + failure reason\n위반 · 미관측 · 범위 이탈 · 대상 상실\n실행 여부는 호출 계층이 결정')
arrow((13.25,4.15),(13.25,3.5));arrow((13.25,3.5),(2.55,3.5));arrow((2.55,3.5),(2.55,2.9));arrow((4.8,2.3),(5.6,2.3));arrow((10.2,2.3),(11,2.3))
ax.text(8,10.5,'상세 계약과 책임 — 제안 아키텍처',ha='center',fontsize=20,weight='bold');ax.text(8,.75,'cuRobo 내부의 TSDF/ESDF 설명은 생략. 보라색: AG3S 의미·상태 / 녹색: 재사용 기하 / 파란색: 궤적 보정·검사',ha='center',fontsize=12)
save(fig,'architecture-detailed')
