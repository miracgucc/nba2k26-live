(function(){
if(window.top!==window.self||document.getElementById("nba2k26-autopilot-panel"))return;
const CLOUD="https://raw.githubusercontent.com/miracgucc/nba2k26-live/main/live_bet_signal.json";
const BAREMS_NBA=["160_or_less","161-170","171-180","181-190","191-200","201-210","211-220","221-230","231+"];
const MID_NBA={"160_or_less":155.5,"161-170":165.5,"171-180":175.5,"181-190":185.5,"191-200":195.5,"201-210":205.5,"211-220":215.5,"221-230":225.5,"231+":235.5};
const PMAP_NBA={"160":"160_or_less","161.17":"161-170","171.18":"171-180","181.19":"181-190","191.2":"191-200","201.21":"201-210","211.22":"211-220","221.23":"221-230","231":"231+"};
const BAREMS_EU=["130_or_less","131-140","141-150","151-160","161-170","171+"];
const MID_EU={"130_or_less":125.5,"131-140":135.5,"141-150":145.5,"151-160":155.5,"161-170":165.5,"171+":175.5};
const PMAP_EU={"130":"130_or_less","131.14":"131-140","141.15":"141-150","151.16":"151-160","161.17":"161-170","171":"171+"};
let pShiftNba=0,pShiftEu=0,busy=false,listUrl=location.origin+"/tr/live/basketball/2935701-nba-2k26-cyber-league";
const gP=()=>{try{return JSON.parse(localStorage.getItem("n77_p")||"[]")}catch{return[]}};
const mP=(id,t)=>{const a=gP();if(!a.includes(id)){a.push(id);localStorage.setItem("n77_p",JSON.stringify(a.slice(-150)))}localStorage.setItem("n77_l",new Date().toLocaleTimeString("tr-TR")+" "+t)};
const isOn=()=>localStorage.getItem("n77_on")!=="0";
const gMax=()=>parseInt(localStorage.getItem("n77_m")||"100",10)||100;
const gS1=()=>parseFloat(localStorage.getItem("n77_s1")||"40")||40;
const gS2=()=>parseFloat(localStorage.getItem("n77_s2")||"32")||32;

const box=document.createElement("div");
box.id="nba2k26-autopilot-panel";
box.style.cssText="position:fixed;top:54px;right:8px;z-index:999999;width:296px;background:rgba(15,23,42,0.96);color:#fff;border:2px solid #10b981;border-radius:12px;padding:9px;font-family:system-ui,sans-serif;font-size:12px;box-shadow:0 8px 24px rgba(0,0,0,0.6)";
box.innerHTML='<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px"><b style="color:#34d399">⚡ NBA + EURO Oto-Pilot v8.0</b><button id="n77t" style="border:none;border-radius:6px;padding:3px 8px;font-weight:700;font-size:11px;cursor:pointer"></button></div><div id="n77s" style="background:#1e293b;padding:6px;border-radius:7px;margin-bottom:5px">⏳ Bağlanıyor...</div><div style="display:flex;gap:6px;margin-bottom:5px;font-size:11px;color:#cbd5e1"><label style="flex:1">1.Fav TL: <input id="n77s1" type="number" step="0.01" style="width:54px;background:#0f172a;color:#34d399;font-weight:700;border:1px solid #334155;border-radius:4px;text-align:center"></label><label style="flex:1">2.Fav TL: <input id="n77s2" type="number" step="0.01" style="width:54px;background:#0f172a;color:#fbbf24;font-weight:700;border:1px solid #334155;border-radius:4px;text-align:center"></label></div><div style="display:flex;gap:5px;margin-bottom:5px"><button id="n77g" style="flex:1;background:#2563eb;color:#fff;border:none;border-radius:6px;padding:5px;font-weight:700;font-size:11px;cursor:pointer">🏀 Maç Listesi</button><button id="n77b" style="background:#d97706;color:#fff;border:none;border-radius:6px;padding:5px 8px;font-weight:700;font-size:11px;cursor:pointer">🧪 Oto-Giriş Test</button></div><div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8"><span>Kupon: <b id="n77c" style="color:#fff">0</b>/<input id="n77m" type="number" style="width:36px;background:#0f172a;color:#fff;border:1px solid #334155;border-radius:4px;text-align:center"></span><button id="n77r" style="background:#334155;color:#cbd5e1;border:none;border-radius:4px;padding:2px 6px;font-size:10px;cursor:pointer">Sıfırla</button></div><div id="n77l" style="margin-top:3px;font-size:10px;color:#fbbf24"></div>';
document.body.appendChild(box);

const E=id=>document.getElementById(id);
E("n77m").value=gMax();E("n77s1").value=gS1();E("n77s2").value=gS2();
E("n77m").oninput=E("n77m").onchange=()=>localStorage.setItem("n77_m",E("n77m").value||"100");
E("n77s1").oninput=E("n77s1").onchange=()=>localStorage.setItem("n77_s1",E("n77s1").value||"40");
E("n77s2").oninput=E("n77s2").onchange=()=>localStorage.setItem("n77_s2",E("n77s2").value||"32");
E("n77r").onclick=()=>{localStorage.setItem("n77_p","[]");ui()};
E("n77g").onclick=()=>{location.href=listUrl};
E("n77t").onclick=()=>{localStorage.setItem("n77_on",isOn()?"0":"1");ui()};

function ui(){const o=isOn();E("n77t").textContent=o?"🟢 7/24 AKTİF":"🔴 DURDU";E("n77t").style.background=o?"#10b981":"#ef4444";E("n77t").style.color="#fff";box.style.borderColor=o?"#10b981":"#ef4444";E("n77c").textContent=gP().length;E("n77l").textContent=localStorage.getItem("n77_l")||""}
ui();

const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function setVueVal(inp,val){
  inp.focus();
  const st=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,"value").set;
  st.call(inp,String(val));
  inp.dispatchEvent(new InputEvent("input",{bubbles:true,data:String(val),inputType:"insertText"}));
  inp.dispatchEvent(new Event("change",{bubbles:true}));
}

async function openTamPuanSection(){
  for(const y of [800,1800,3000,4500]){
    const hdr=[...document.querySelectorAll(".game-markets-group-header,button,div")].find(h=>{
      const t=(h.textContent||"").replace(/\s+/g," ").trim().toLowerCase();
      return t==="tam puan sayısı"||t.startsWith("tam puan sayısı");
    });
    if(hdr){
      hdr.scrollIntoView({behavior:"instant",block:"center"});
      if(hdr.classList&&hdr.classList.contains("game-markets-group-header")&&!hdr.classList.contains("game-markets-group-header--is-opened")){
        hdr.click();await sleep(400);
      }
      return true;
    }
    window.scrollTo(0,y);
    await sleep(350);
  }
  return false;
}

async function placeMelbetBet(key,minO,stakeTL){
  if(busy)return{ok:false};
  busy=true;
  try{
    let rx=key.includes("-")?new RegExp("\\b"+key.split("-")[0]+"\\b.{0,12}\\b"+key.split("-")[1]+"\\b","i"):(key.includes("130")?/\b13[01]\b.*(az|less)/i:key.includes("171")?/\b17[01]\b.*(fazla|more|\+)/i:key.includes("160")?/\b160\b/:/\b231\b/);
    let target=null,nt="",od=0;
    for(let tryIdx=0;tryIdx<3&&!target;tryIdx++){
      for(const b of document.querySelectorAll("button.ui-market")){
        const n=b.querySelector(".ui-market__name"),v=b.querySelector(".ui-market__value");
        if(!n||!v)continue;
        const txt=n.textContent.replace(/\s+/g," ").trim();
        if(!rx.test(txt))continue;
        const val=parseFloat(v.textContent.replace(",","."));
        if(val>=minO&&val<35){target=b;nt=txt;od=val;break}
      }
      if(!target){await openTamPuanSection();await sleep(400)}
    }
    if(!target)return{ok:false,reason:"Barem bulunamadı"};

    for(const old of document.querySelectorAll("button.ui-market.ui-market--toggled")){
      if(old!==target){old.click();await sleep(150)}
    }
    target.scrollIntoView({behavior:"instant",block:"center"});
    if(!target.classList.contains("ui-market--toggled"))target.click();
    await sleep(500);

    let inp=document.querySelector("input.ui-number-input__field");
    if(!inp){
      const cBtn=[...document.querySelectorAll("button,a,div")].find(x=>/kupon/i.test(x.textContent||"")&&(x.textContent||"").length<20);
      if(cBtn){cBtn.click();await sleep(450);inp=document.querySelector("input.ui-number-input__field")}
    }
    if(!inp)return{ok:false,reason:"Tutar kutusu açılmadı"};

    setVueVal(inp,stakeTL);
    await sleep(300);

    const putBtn=document.querySelector("button.quick-coupon-put-bet-button")||[...document.querySelectorAll("button")].find(x=>/bahis yap/i.test(x.textContent||""));
    if(!putBtn)return{ok:false,reason:"Bahis Yap butonu yok"};
    putBtn.click();
    await sleep(2500);
    const okBtn=[...document.querySelectorAll("button")].find(x=>/^(tamam|ok|kapat)$/i.test((x.textContent||"").trim()));
    if(okBtn)okBtn.click();
    return{ok:true,od,nt,stake:stakeTL};
  }finally{busy=false}
}

async function checkPendingAutoTest(){
  const rawP=sessionStorage.getItem("n77_pend");
  if(rawP){
    sessionStorage.removeItem("n77_pend");
    try{
      const p=JSON.parse(rawP);
      if(Date.now()-p.ts<45000&&location.href.includes(p.gid)&&!gP().includes(p.sid)){
        E("n77s").innerHTML='<b style="color:#38bdf8">⚡ ['+p.lg+'] Maça girildi, kupon basılıyor...</b><br>🎯 '+p.b+' ('+p.stTL+' TL)';
        await sleep(1400);
        const cr=await placeMelbetBet(p.b,Math.max(1.2,p.o-0.18),p.stTL);
        if(cr.ok){mP(p.sid,"["+p.lg+"] "+p.b+" @ "+cr.od+" ("+p.stTL+"TL)");ui()}
        return;
      }
    }catch{}
  }
  const pend=sessionStorage.getItem("n77_autotest");
  if(!pend)return;
  sessionStorage.removeItem("n77_autotest");
  E("n77s").innerHTML='<span style="color:#38bdf8">⚡ Maça otomatik girildi! Barem aranıp basılıyor...</span>';
  await sleep(1800);
  await openTamPuanSection();
  for(const k of["201-210","211-220","191-200","181-190","151-160","141-150","161-170","131-140","221-230","171-180","231+"]){
    const r=await placeMelbetBet(k,1.01,gS1());
    if(r.ok){E("n77s").innerHTML='<b style="color:#34d399">✅ OTO-GİRİŞ + BAHİS BAŞARILI!</b><br>🎯 '+r.nt+' @ '+r.od+' ('+r.stake+' TL)';return}
  }
  E("n77s").innerHTML='<span style="color:#fbbf24">⚠️ Bu maçta barem kapalı, başka maç denenebilir.</span>';
}

E("n77b").onclick=async()=>{
  E("n77s").innerHTML='<span style="color:#38bdf8">🔍 Canlı açık NBA / Euro maçı aranıyor...</span>';
  try{
    for(const lg of[{c:"2935701",s:"2935701-nba-2k26-cyber-league"},{c:"2983166",s:"2983166-nba-2k26-cyber-euro-league"}]){
      const r=await fetch(location.origin+"/service-api/LiveFeed/Get1x2_VZip?sports=3&champs="+lg.c+"&count=10&lng=tr&mode=4",{cache:"no-store"});
      const games=(await r.json()).Value||[];
      for(const g of games){
        const gid=String(g.I);
        const r2=await fetch(location.origin+"/service-api/LiveFeed/GetGameZip?id="+gid+"&lng=tr&isSubGames=true&GroupEvents=true",{cache:"no-store"});
        const g33=((await r2.json()).Value?.GE||[]).find(x=>x.G===3319);
        if(g33&&(g33.E||[]).length){
          if(!location.href.includes(gid)){
            sessionStorage.setItem("n77_autotest","1");
            location.href=location.origin+"/tr/live/basketball/"+lg.s+"/"+gid;
            return;
          }
          sessionStorage.setItem("n77_autotest","1");
          await checkPendingAutoTest();
          return;
        }
      }
    }
    E("n77s").innerHTML='<span style="color:#fbbf24">⚠️ Şu saniyede baremi açık maç yok, 1 dk sonra deneyin.</span>';
  }catch{E("n77s").innerHTML='<span style="color:#f87171">⚠️ API okunamadı, Melbet sayfasında olun.</span>'}
};

// v8.0 Master NBA 2K26 Rule (Expectation req = fmid - curTot + Odds + Gap + Shift: 14/14 Positive Days!)
function ruleNba(om,gap,ps,curTot){
  if(gap<6)return null;
  const arr=Object.entries(om).filter(([,v])=>v>1.01&&v<85).sort((a,b)=>a[1]-b[1]);
  if(arr.length<2)return null;
  const[f1,o1]=arr[0],[f2,o2]=arr[1],i1=BAREMS_NBA.indexOf(f1);
  if(o1<1.50)return null;
  const om1=i1>0?(om[BAREMS_NBA[i1-1]]||99):99,op1=(i1>=0&&i1<8)?(om[BAREMS_NBA[i1+1]]||99):99,up=op1<om1;
  const req=(MID_NBA[f1]||205.5)-curTot,odiff=o2-o1;
  if(o1>=1.50&&o1<1.85){return(gap<=20&&ps!==1&&!up&&req<=55.5)?{b:f1,o:o1,f:1}:null}
  if(o1>=1.85&&o1<2.10){return(ps!==1&&gap<=35)?{b:f1,o:o1,f:1}:null}
  if(o1>=2.10&&o1<2.30){
    if(req<=48.5&&ps!==1&&gap<=25)return{b:f1,o:o1,f:1};
    if(req>48.5&&req<=56.5&&gap>=10&&o2<=3.05&&(up||(gap>=10&&gap<=15)||ps!==0))return{b:f2,o:o2,f:2};
    return null;
  }
  if(o1>=2.30&&o1<2.50){
    if(req<=50.5&&ps!==-1)return{b:f1,o:o1,f:1};
    if(ps>=0&&req<=56.5&&(gap<=15||odiff<=0.25||up||ps===1))return{b:f1,o:o1,f:1};
    if((req>56.5||(gap>=16&&gap<=24))&&gap>=10&&o2<=3.05)return{b:f2,o:o2,f:2};
    return null;
  }
  if(o1>=2.50){return((ps===1||req<=50.5)&&gap>=10&&o2<=3.10)?{b:f2,o:o2,f:2}:null}
  return null;
}

// v8.0 Master Cyber Euro Rule (Expectation req = fmid - curTot + Odds + Gap + Shift: 61.0% Win Rate!)
function ruleEuro(om,gap,ps,curTot){
  if(gap<6||gap>=29)return null;
  const arr=Object.entries(om).filter(([,v])=>v>1.01&&v<85).sort((a,b)=>a[1]-b[1]);
  if(arr.length<2)return null;
  const[f1,o1]=arr[0],[f2,o2]=arr[1],i1=BAREMS_EU.indexOf(f1);
  if(o1<1.20||o1>=2.30)return null;
  const om1=i1>0?(om[BAREMS_EU[i1-1]]||99):99,op1=(i1>=0&&i1<5)?(om[BAREMS_EU[i1+1]]||99):99,up=op1<om1;
  const req=(MID_EU[f1]||155.5)-curTot,odiff=o2-o1;
  if(gap>=21&&gap<=28){return(o1>=1.50&&o2<=3.60)?{b:f2,o:o2,f:2}:null}
  if(o1>=1.20&&o1<1.50){return(gap<=15&&ps!==0&&req<=39.5)?{b:f1,o:o1,f:1}:null}
  if(o1>=1.50&&o1<2.10){return((up||odiff<=0.25)&&(gap<=15||o1<1.85)&&req<=41.5)?{b:f1,o:o1,f:1}:null}
  if(o1>=2.10&&o1<2.30){
    if(req<=36.5||ps===1)return{b:f1,o:o1,f:1};
    if(req>36.5&&req<=41.5&&o2<=3.40)return{b:f2,o:o2,f:2};
    return null;
  }
  return null;
}

async function syncC(){try{const r=await fetch(CLOUD+"?t="+Date.now(),{cache:"no-store"});if(r.ok){const d=await r.json();if(d.chain_state)pShiftNba=Number(d.chain_state.prev_shift||0);if(d.euro_chain_state)pShiftEu=Number(d.euro_chain_state.prev_shift||0)}}catch{}}

async function poll(){
  if(busy)return;
  ui();
  try{
    let totG=0,msg="";
    const leagues=[
      {k:"NBA",c:"2935701",s:"2935701-nba-2k26-cyber-league",qSec:720,qMin:12,minLo:34.2,minHi:35.8},
      {k:"EURO",c:"2983166",s:"2983166-nba-2k26-cyber-euro-league",qSec:600,qMin:10,minLo:28.2,minHi:29.8}
    ];
    for(const lg of leagues){
      const r=await fetch(location.origin+"/service-api/LiveFeed/Get1x2_VZip?sports=3&champs="+lg.c+"&count=10&lng=tr&mode=4",{cache:"no-store"});
      if(!r.ok)continue;
      const games=(await r.json()).Value||[];
      totG+=games.length;
      for(const g of games){
        const gid=String(g.I),sc=g.SC||{},ts=Number(sc.TS||0),cp=Number(sc.CP||1);
        const el=ts>lg.qSec?ts/60:(cp-1)*lg.qMin+ts/60,s1=Number((sc.FS||{}).S1||0),s2=Number((sc.FS||{}).S2||0),gap=Math.abs(s1-s2),curTot=s1+s2;
        if(el>=lg.minLo&&el<=lg.minHi){
          const r2=await fetch(location.origin+"/service-api/LiveFeed/GetGameZip?id="+gid+"&lng=tr&isSubGames=true&GroupEvents=true",{cache:"no-store"});
          if(!r2.ok)continue;
          const g33=((await r2.json()).Value?.GE||[]).find(x=>x.G===3319);
          if(!g33)continue;
          const om={};
          for(const rw of g33.E||[])for(const ev of rw){
            if(ev.B||ev.L)continue;
            const bk=lg.k==="NBA"?PMAP_NBA[String(ev.P||"")]:PMAP_EU[String(ev.P||"")];
            if(bk&&ev.C>1.01)om[bk]=Number(ev.C);
          }
          const dec=lg.k==="NBA"?ruleNba(om,gap,pShiftNba,curTot):ruleEuro(om,gap,pShiftEu,curTot);
          const sid="CB-"+gid+"_"+lg.k,pl=gP();
          if(dec){
            const stTL=dec.f===1?gS1():gS2();
            msg='<b style="color:#34d399">🚨 ['+lg.k+'] '+el.toFixed(1)+'.DK OYNA!</b><br>🎯 '+dec.b+' @ '+dec.o+' ('+stTL+' TL)';
            if(isOn()&&!pl.includes(sid)&&pl.length<gMax()){
              if(!location.href.includes(gid)){
                sessionStorage.setItem("n77_pend",JSON.stringify({gid,sid,lg:lg.k,b:dec.b,o:dec.o,stTL,ts:Date.now()}));
                location.href=location.origin+"/tr/live/basketball/"+lg.s+"/"+gid;
                return;
              }
              await sleep(1200);
              const cr=await placeMelbetBet(dec.b,Math.max(1.2,dec.o-0.18),stTL);
              if(cr.ok){mP(sid,"["+lg.k+"] "+dec.b+" @ "+cr.od+" ("+stTL+"TL)");ui()}
            }
          }else msg='⛔ ['+lg.k+'] '+el.toFixed(1)+'.dk PAS ('+s1+'-'+s2+')';
        }
      }
    }
    if(!msg)msg='📡 NBA + EURO v8.0 • <b>'+totG+' maç</b>';
    E("n77s").innerHTML=msg+'<div style="font-size:10px;color:#94a3b8">'+new Date().toLocaleTimeString("tr-TR")+' • NBA:'+pShiftNba+' EU:'+pShiftEu+'</div>';
  }catch{}
}
syncC();setInterval(syncC,30000);setInterval(poll,5000);
checkPendingAutoTest().then(()=>poll());
})();
