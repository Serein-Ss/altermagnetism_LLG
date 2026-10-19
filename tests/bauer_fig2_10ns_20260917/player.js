'use strict';
const payloads=JSON.parse(document.getElementById('paths').textContent);
document.querySelectorAll('.figure').forEach(im=>im.addEventListener('click',()=>{
  const modal=document.getElementById('zoom');modal.querySelector('img').src=im.src;modal.showModal();
}));
document.querySelectorAll('.player').forEach((root,b)=>{
  const p=payloads[b], bytes=Uint8Array.from(atob(p.values),c=>c.charCodeAt(0));
  const values=new DataView(bytes.buffer), n=p.time_ns.length, sites=p.shape[1];
  const spin=(k,i,c)=>values.getInt16(2*((k*sites+i)*3+c),true)/32767;
  const history=root.querySelector('.history'), state=root.querySelector('.state');
  const hc=history.getContext('2d'), sc=state.getContext('2d');
  const bg=document.createElement('canvas');bg.width=1200;bg.height=400;const ctx=bg.getContext('2d');
  const scrub=root.querySelector('.scrub'), start=root.querySelector('.start'), end=root.querySelector('.end');
  const play=root.querySelector('.play'), speed=root.querySelector('.speed');
  let k=0, lo=0, hi=n-1, playing=false, previous=null, clockTime=0;
  const left=65, right=1180, width=right-left;
  const index=t=>Math.max(0,Math.min(n-1,Math.round(t/(p.time_ns[1]-p.time_ns[0]))));
  const x=t=>left+(t-p.time_ns[lo])/(p.time_ns[hi]-p.time_ns[lo])*width;
  function background(){
    ctx.fillStyle='white';ctx.fillRect(0,0,1200,400);ctx.font='16px sans-serif';ctx.fillStyle='#182534';
    ctx.fillText('Total Mz (100 spins)',left,20);
    ctx.strokeStyle='#ddd';ctx.lineWidth=1;
    for(const m of [-100,0,100]){const y=100-m*.6;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillText(m.toString(),12,y+5);}
    ctx.strokeStyle='#163b8f';ctx.beginPath();
    for(let j=lo;j<=hi;j++){const xx=x(p.time_ns[j]),yy=100-p.mz[j]*.6;if(j===lo)ctx.moveTo(xx,yy);else ctx.lineTo(xx,yy);}ctx.stroke();
    ctx.fillStyle='#182534';ctx.fillText('Space-time map: site 99 at top, site 0 at bottom',left,200);
    const raster=ctx.createImageData(width,140);
    for(let xx=0;xx<width;xx++){
      const j=Math.round(lo+xx/(width-1)*(hi-lo));
      for(let yy=0;yy<140;yy++){
        const i=99-Math.floor(yy/140*100), pos=4*(yy*width+xx), positive=spin(j,i,2)>=0;
        raster.data[pos]=positive?0:16;raster.data[pos+1]=positive?201:57;raster.data[pos+2]=positive?123:232;raster.data[pos+3]=255;
      }
    }
    ctx.putImageData(raster,left,210);ctx.fillStyle='#182534';
    for(let a=0;a<=5;a++){const t=p.time_ns[lo]+a/5*(p.time_ns[hi]-p.time_ns[lo]);ctx.fillText(t.toFixed(3),left+a/5*width-18,377);}
    ctx.fillText('Time (ns)',550,398);
  }
  function color(z){const a=Math.abs(z);const target=z>=0?[0,155,90]:[20,65,215];return `rgb(${target.map(v=>Math.round(245+(v-245)*a)).join(',')})`;}
  function draw(){
    hc.drawImage(bg,0,0);hc.strokeStyle='#ef2222';hc.lineWidth=2;hc.beginPath();hc.moveTo(x(p.time_ns[k]),30);hc.lineTo(x(p.time_ns[k]),350);hc.stroke();
    sc.fillStyle='white';sc.fillRect(0,0,1200,290);sc.fillStyle='#182534';sc.font='16px sans-serif';
    sc.fillText('Instantaneous chain: color = sz; no arrows or artificial smoothing',left,22);
    sc.strokeStyle='#aaa';sc.beginPath();sc.moveTo(left,52);sc.lineTo(right,52);sc.stroke();
    for(let i=0;i<sites;i++){sc.fillStyle=color(spin(k,i,2));sc.beginPath();sc.arc(left+i/(sites-1)*width,52,5,0,2*Math.PI);sc.fill();}
    sc.fillStyle='#182534';sc.fillText('Local sz profile: domain boundaries appear where the profile crosses zero',left,96);
    for(const z of [-1,0,1]){const y=179-z*61;sc.strokeStyle=z===0?'#999':'#eee';sc.beginPath();sc.moveTo(left,y);sc.lineTo(right,y);sc.stroke();sc.fillText(z.toString(),20,y+5);}
    sc.strokeStyle='#777';sc.beginPath();for(let i=0;i<sites;i++){const xx=left+i/(sites-1)*width,yy=179-spin(k,i,2)*61;if(i===0)sc.moveTo(xx,yy);else sc.lineTo(xx,yy);}sc.stroke();
    for(let i=0;i<sites;i++){sc.fillStyle=color(spin(k,i,2));sc.beginPath();sc.arc(left+i/(sites-1)*width,179-spin(k,i,2)*61,3,0,2*Math.PI);sc.fill();}
    sc.fillStyle='#182534';for(const i of [0,20,40,60,80,99])sc.fillText(String(i),left+i/99*width-8,266);sc.fillText('Chain site',550,287);
    scrub.value=k;root.querySelector('.readout').textContent=`t=${p.time_ns[k].toFixed(3)} ns · 总Mz=${p.mz[k].toFixed(2)} · 播放器帧 ${k+1}/${n} · 蓝色负向 / 绿色正向 / 白色近零`;
  }
  function stop(){playing=false;play.textContent='播放';previous=null;}
  function setRange(a,z){
    if(!Number.isFinite(a)||!Number.isFinite(z)||a<0||z>10||z-a<.001){alert('请设置0–10 ns内、至少0.001 ns的时间窗口');return;}
    stop();lo=index(a);hi=index(z);if(hi<=lo)hi=Math.min(n-1,lo+1);
    start.value=p.time_ns[lo].toFixed(3);end.value=p.time_ns[hi].toFixed(3);scrub.min=lo;scrub.max=hi;k=lo;background();draw();
  }
  function tick(now){if(!playing)return;if(previous!==null){clockTime+=(now-previous)/1000*Number(speed.value);k=index(clockTime);if(k>=hi){k=hi;draw();stop();return;}}previous=now;draw();requestAnimationFrame(tick);}
  play.onclick=()=>{if(playing){stop();return;}if(k>=hi)k=lo;clockTime=p.time_ns[k];previous=null;playing=true;play.textContent='暂停';requestAnimationFrame(tick);};
  scrub.oninput=()=>{stop();k=Number(scrub.value);draw();};
  root.querySelector('.back').onclick=()=>{stop();k=Math.max(lo,k-1);draw();};
  root.querySelector('.next').onclick=()=>{stop();k=Math.min(hi,k+1);draw();};
  root.querySelector('.range').onclick=()=>setRange(Number(start.value),Number(end.value));
  root.querySelector('.all').onclick=()=>setRange(0,10);
  root.querySelector('.detail').onclick=()=>{speed.value='.01';setRange(Math.max(0,p.detail_center_ns-.25),Math.min(10,p.detail_center_ns+.25));};
  root.querySelector('.expand').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else root.requestFullscreen().catch(()=>alert('可使用浏览器缩放查看'));};
  setRange(0,10);
});
