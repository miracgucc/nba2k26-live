(function(){
if(window.top!==window.self||document.getElementById("nba2k26-autopilot-panel"))return;
const TG_TOK="__TG_TOKEN__";
const DEFAULT_CID="1244842635";
const CLOUD="https://raw.githubusercontent.com/miracgucc/nba2k26-live/main/live_bet_signal.json";
const BAREMS_NBA=["160_or_less","161-170","171-180","181-190","191-200","201-210","211-220","221-230","231+"];
const MID_NBA={"160_or_less":155.5,"161-170":165.5,"171-180":175.5,"181-190":185.5,"191-200":195.5,"201-210":205.5,"211-220":215.5,"221-230":225.5,"231+":235.5};
const PMAP_NBA={"160":"160_or_less","161.17":"161-170","171.18":"171-180","181.19":"181-190","191.2":"191-200","201.21":"201-210","211.22":"211-220","221.23":"221-230","231":"231+"};
const BAREMS_EU=["130_or_less","131-140","141-150","151-160","161-170","171+"];
const MID_EU={"130_or_less":125.5,"131-140":135.5,"141-150":145.5,"151-160":155.5,"161-170":165.5,"171+":175.5};
const PMAP_EU={"130":"130_or_less","131.14":"131-140","141.15":"141-150","151.16":"151-160","161.17":"161-170","171":"171+"};

let pShiftNba=Number(localStorage.getItem("n80_ps_nba")||"0")||0;
let pShift27=Number(localStorage.getItem("n80_ps_27")||"0")||0;
let pShiftEu=Number(localStorage.getItem("n80_ps_eu")||"0")||0;
let busy=false,listUrl=location.origin+"/tr/live/basketball/2935701-nba-2k26-cyber-league";

if(!localStorage.getItem("n81_init3000")){
  localStorage.setItem("n77_s1","150");
  localStorage.setItem("n77_s2","120");
  localStorage.setItem("n77_m","100");
  localStorage.setItem("n77_p","[]");
  localStorage.setItem("n77_on","1");
  localStorage.setItem("n81_init3000","1");
}
if(!localStorage.getItem("n78_cid"))localStorage.setItem("n78_cid",DEFAULT_CID);

const jGet=(k,d)=>{try{return JSON.parse(localStorage.getItem(k)||JSON.stringify(d))}catch{return d}};
const jSet=(k,v)=>localStorage.setItem(k,JSON.stringify(v));
const gP=()=>jGet("n77_p",[]);
const mP=(id,t)=>{const a=gP();if(!a.includes(id)){a.push(id);jSet("n77_p",a.slice(-150))}localStorage.setItem("n77_l",new Date().toLocaleTimeString("tr-TR")+" "+t)};
const isOn=()=>localStorage.getItem("n77_on")!=="0";
const gMax=()=>parseInt(localStorage.getItem("n77_m")||"100",10)||100;
const gS1=()=>parseFloat(localStorage.getItem("n77_s1")||"150")||150;
const gS2=()=>parseFloat(localStorage.getItem("n77_s2")||"120")||120;
const getCid=()=>localStorage.getItem("n78_cid")||DEFAULT_CID;
const gStats=()=>jGet("n78_stats",{w:0,l:0,net:0,hist:[]});
const gOpen=()=>jGet("n78_open",[]);
const gPas=()=>jGet("n80_pas",[]);
const mPas=id=>{const a=gPas();if(!a.includes(id)){a.push(id);jSet("n80_pas",a.slice(-150))}};
const isPasTgOn=()=>localStorage.getItem("n80_pastg")!=="0";

function readWallet(){
  const el=document.querySelector(".balance__wallet")||document.querySelector("[class*='balance__wallet']");
  const raw=(el?el.textContent:"").replace(/\s+/g," ").trim();
  const m=raw.match(/[\d]+([.,]\d+)?/);
  const num=m?parseFloat(m[0].replace(",",".")):null;
  return{txt:raw||"Melbet Açık",num};
}

function getTier1500(b){
  if(!b||b<4200)return{k:1,s1:150,s2:120};
  if(b<5700)return{k:2,s1:210,s2:170};
  if(b<7600)return{k:3,s1:280,s2:220};
  if(b<10400)return{k:4,s1:380,s2:300};
  if(b<14200)return{k:5,s1:520,s2:420};
  if(b<19400)return{k:6,s1:710,s2:570};
  return{k:7,s1:850,s2:680};
}

function scoreToBarem(tot,lg){
  if(lg==="EURO"){
    if(tot<=130)return"130_or_less";
    if(tot<=140)return"131-140";
    if(tot<=150)return"141-150";
    if(tot<=160)return"151-160";
    if(tot<=170)return"161-170";
    return"171+";
  }
  if(tot<=160)return"160_or_less";
  if(tot<=170)return"161-170";
  if(tot<=180)return"171-180";
  if(tot<=190)return"181-190";
  if(tot<=200)return"191-200";
  if(tot<=210)return"201-210";
  if(tot<=220)return"211-220";
  if(tot<=230)return"221-230";
  return"231+";
}

async function sendTg(text,cidOverride){
  const cid=cidOverride||getCid();
  if(!cid||TG_TOK.includes("__TG_"))return false;
  try{
    const r=await fetch("https://api.telegram.org/bot"+TG_TOK+"/sendMessage",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({chat_id:cid,text,parse_mode:"HTML",disable_web_page_preview:true})
    });
    return r.ok;
  }catch{return false}
}

const box=document.createElement("div");
box.id="nba2k26-autopilot-panel";
box.style.cssText="position:fixed;top:54px;right:8px;z-index:999999;width:298px;background:rgba(15,23,42,0.96);color:#fff;border:2px solid #10b981;border-radius:12px;padding:9px;font-family:system-ui,sans-serif;font-size:12px;box-shadow:0 8px 24px rgba(0,0,0,0.6)";
box.innerHTML='<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px"><b style="color:#34d399">⚡ NBA + EURO v8.1 + TG</b><button id="n77t" style="border:none;border-radius:6px;padding:3px 8px;font-weight:700;font-size:11px;cursor:pointer"></button></div><div id="n77s" style="background:#1e293b;padding:6px;border-radius:7px;margin-bottom:5px">⏳ Bağlanıyor...</div><div style="display:flex;gap:6px;margin-bottom:5px;font-size:11px;color:#cbd5e1"><label style="flex:1">1.Fav TL: <input id="n77s1" type="number" step="0.01" style="width:54px;background:#0f172a;color:#34d399;font-weight:700;border:1px solid #334155;border-radius:4px;text-align:center"></label><label style="flex:1">2.Fav TL: <input id="n77s2" type="number" step="0.01" style="width:54px;background:#0f172a;color:#fbbf24;font-weight:700;border:1px solid #334155;border-radius:4px;text-align:center"></label></div><div style="display:flex;gap:4px;margin-bottom:5px"><button id="n77g" style="flex:1;background:#2563eb;color:#fff;border:none;border-radius:6px;padding:5px;font-weight:700;font-size:11px;cursor:pointer">🏀 Liste</button><button id="n78tg" style="flex:1;background:#0284c7;color:#fff;border:none;border-radius:6px;padding:5px;font-weight:700;font-size:11px;cursor:pointer">📲 TG Test</button><button id="n77b" style="background:#d97706;color:#fff;border:none;border-radius:6px;padding:5px 6px;font-weight:700;font-size:11px;cursor:pointer">🧪 Test</button></div><div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8"><span>Kupon: <b id="n77c" style="color:#fff">0</b>/<input id="n77m" type="number" style="width:36px;background:#0f172a;color:#fff;border:1px solid #334155;border-radius:4px;text-align:center"></span><span id="n78st" style="color:#38bdf8"></span><button id="n77r" style="background:#334155;color:#cbd5e1;border:none;border-radius:4px;padding:2px 6px;font-size:10px;cursor:pointer">Sıfırla</button></div><div id="n77l" style="margin-top:3px;font-size:10px;color:#fbbf24"></div>';
document.body.appendChild(box);

const E=id=>document.getElementById(id);
function ui(){
  const o=isOn(),st=gStats();
  if(document.activeElement!==E("n77m"))E("n77m").value=gMax();
  if(document.activeElement!==E("n77s1"))E("n77s1").value=gS1();
  if(document.activeElement!==E("n77s2"))E("n77s2").value=gS2();
  E("n77t").textContent=o?"🟢 7/24 AKTİF":"🔴 DURDU";
  E("n77t").style.background=o?"#10b981":"#ef4444";
  E("n77t").style.color="#fff";
  box.style.borderColor=o?"#10b981":"#ef4444";
  E("n77c").textContent=gP().length;
  E("n78st").textContent=st.w+"K/"+st.l+"Y ("+(st.net>=0?"+":"")+st.net.toFixed(0)+"₺)";
  E("n77l").textContent="📲 TG Aktif • "+(localStorage.getItem("n77_l")||"Nöbette");
}
E("n77m").oninput=E("n77m").onchange=()=>{const v=parseInt(E("n77m").value,10);if(v>0)localStorage.setItem("n77_m",String(v))};
E("n77s1").oninput=E("n77s1").onchange=()=>{const v=parseFloat(E("n77s1").value);if(v>0)localStorage.setItem("n77_s1",String(v))};
E("n77s2").oninput=E("n77s2").onchange=()=>{const v=parseFloat(E("n77s2").value);if(v>0)localStorage.setItem("n77_s2",String(v))};
E("n77r").onclick=()=>{jSet("n77_p",[]);ui()};
E("n77g").onclick=()=>{location.href=listUrl};
E("n77t").onclick=()=>{localStorage.setItem("n77_on",isOn()?"0":"1");ui()};
E("n78tg").onclick=async()=>{
  const ok=await sendTg(buildStatusMsg("📲 <b>NBA + EURO v8.1 Telegram Testi Başarılı!</b>"));
  E("n77l").textContent=ok?"✅ Telegram mesajı gönderildi!":"⚠️ Mesaj gönderilemedi";
};
ui();

function buildStatusMsg(hdr){
  const w=readWallet(),st=gStats(),op=gOpen(),tr=getTier1500(w.num);
  let m=(hdr||"📊 <b>NBA 2K26 + EURO v8.1 Kasa Raporu</b>")+"\n\n";
  m+="💰 <b>Melbet Bakiye:</b> "+w.txt+"\n";
  m+="⚙️ <b>Mod:</b> "+(isOn()?"🟢 7/24 AKTİF":"🔴 DURDURULDU")+"\n";
  m+="💵 <b>Aktif Tutar:</b> 1.Fav <b>"+gS1()+" TL</b> | 2.Fav <b>"+gS2()+" TL</b>\n";
  if(w.num)m+="📈 <b>3.000 TL Kademe Planı:</b> Kademe "+tr.k+" ("+tr.s1+" / "+tr.s2+" TL)\n";
  m+="🏆 <b>Toplam Sonuç:</b> "+st.w+" Kazandı / "+st.l+" Kaybetti\n";
  m+="📈 <b>Toplam Net:</b> <b>"+(st.net>=0?"+":"")+st.net.toFixed(1)+" TL</b>\n";
  if(op.length){
    m+="\n⏳ <b>Sonuç Bekleyen Kupon ("+op.length+"):</b>\n";
    for(const b of op)m+="• ["+b.lg+"] "+b.teams+" | "+b.barem+" @ "+b.odd+" ("+b.stake+" TL)\n";
  }
  if(st.hist&&st.hist.length){
    m+="\n📝 <b>Son Sonuçlananlar:</b>\n";
    for(const h of st.hist.slice(-5).reverse())m+="• "+h+"\n";
  }
  return m;
}

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

async function recordAndNotifyBet(p,cr){
  mP(p.sid,"["+p.lg+"] "+p.b+" @ "+cr.od+" ("+p.stTL+"TL)");
  const op=gOpen();
  if(!op.some(x=>x.gid===p.gid)){
    op.push({gid:p.gid,lg:p.lg,teams:p.teams||p.gid,sc:p.sc||"-",favType:p.f===1?"1. Favori":"2. Favori",barem:p.b,odd:cr.od,stake:p.stTL,ts:Date.now()});
    jSet("n78_open",op);
  }
  ui();
  const w=readWallet();
  await sendTg(
    "🚨 <b>YENİ BAHİS ALINDI! ["+p.lg+"]</b>\n\n"+
    "🏀 <b>"+(p.teams||p.gid)+"</b>\n"+
    "⏱️ Karar Skoru: <code>"+(p.sc||"-")+"</code>\n"+
    "🎯 Seçim: <b>"+(p.f===1?"1. Favori":"2. Favori")+" ("+p.b+")</b>\n"+
    "📈 Oran: <b>"+cr.od+"</b> | Tutar: <b>"+p.stTL+" TL</b>\n"+
    "💰 Melbet Bakiye: "+w.txt
  );
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
        if(cr.ok)await recordAndNotifyBet(p,cr);
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
    if(r.ok){
      E("n77s").innerHTML='<b style="color:#34d399">✅ OTO-GİRİŞ + BAHİS BAŞARILI!</b><br>🎯 '+r.nt+' @ '+r.od+' ('+r.stake+' TL)';
      await sendTg("🧪 <b>Test Bahsi Basıldı:</b> "+r.nt+" @ "+r.od+" ("+r.stake+" TL)");
      return;
    }
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

function ruleNba(om,gap,ps,curTot){
  const arr=Object.entries(om).filter(([,v])=>v>1.01&&v<85).sort((a,b)=>a[1]-b[1]);
  if(arr.length<2)return null;
  const[f1,o1]=arr[0],[f2,o2]=arr[1],i1=BAREMS_NBA.indexOf(f1),i2=BAREMS_NBA.indexOf(f2);
  const om1=i1>0?(om[BAREMS_NBA[i1-1]]||99):99,op1=(i1>=0&&i1<8)?(om[BAREMS_NBA[i1+1]]||99):99,up=op1<om1,f2Up=i2>i1;
  const req=(MID_NBA[f1]||205.5)-curTot,odiff=o2-o1;
  const pas=r=>({play:false,reason:r,f1,o1,f2,o2,req});
  if(gap<6)return pas("Skor farkı çok yakın ("+gap+" sayı < 6, uzatma riski)");
  if(o1<1.50)return pas("1.Favori oranı çok düşük ("+o1+" < 1.50)");
  if(o1>=1.50&&o1<1.85){
    return(gap<=24&&ps!==1&&!f2Up&&req<=59.5)?{play:true,b:f1,o:o1,f:1,req}:pas("1.50-1.85 filtresi (Fark:"+gap+", 1.Fav İstediği:"+req.toFixed(1)+")");
  }
  if(o1>=1.85&&o1<2.10){
    if(ps!==1&&gap<=32){
      if(gap>=20&&f2Up&&req<47.5)return pas("1.85-2.10 yüksek fark + düşük barem riski (Fark:"+gap+", İstenen:"+req.toFixed(1)+")");
      return{play:true,b:f1,o:o1,f:1,req};
    }
    return pas("1.85-2.10 filtresi (Önceki maç üst kayma veya fark > 32)");
  }
  if(o1>=2.10&&o1<2.30){
    if(gap>=25&&f2Up&&req<=53.5)return{play:true,b:f1,o:o1,f:1,req};
    if(req<=48.5&&ps!==1&&gap<=24)return{play:true,b:f1,o:o1,f:1,req};
    if(req>48.5&&req<=56.5&&gap>=10&&gap<=24&&o2<=3.05){
      if(gap>=16&&f2Up&&req<=50.5)return{play:true,b:f1,o:o1,f:1,req};
      if(up||(gap>=10&&gap<=15)||!f2Up||ps!==0)return{play:true,b:f2,o:o2,f:2,req};
    }
    return pas("2.10-2.30 beklenti dışı (İstenen:"+req.toFixed(1)+", Fark:"+gap+", 2.Fav:"+o2+")");
  }
  if(o1>=2.30&&o1<2.50){
    if(gap>=16&&gap<=28&&!f2Up&&req>=51.5&&o2<=3.10)return{play:true,b:f2,o:o2,f:2,req};
    if(req<=50.5&&ps!==-1&&gap<=24)return{play:true,b:f1,o:o1,f:1,req};
    if(gap<=15&&ps>=0&&req<=56.5&&(odiff<=0.25||up||ps===1||req<=54.5))return{play:true,b:f1,o:o1,f:1,req};
    if(req>56.5&&gap>=10&&gap<=24&&!f2Up&&o2<=3.05)return{play:true,b:f2,o:o2,f:2,req};
    return pas("2.30-2.50 beklenti dışı (İstenen:"+req.toFixed(1)+", Fark:"+gap+", 2.Fav:"+o2+")");
  }
  if(o1>=2.50){
    return((ps===1||req<=50.5)&&gap>=10&&gap<=18&&o2<=3.10)?{play:true,b:f2,o:o2,f:2,req}:pas("2.50+ yüksek risk (İstenen:"+req.toFixed(1)+", Fark:"+gap+", 2.Fav:"+o2+")");
  }
  return pas("Kural dışı oran yapısı");
}

function ruleNba27(om,gap,ps,curTot){
  const arr=Object.entries(om).filter(([,v])=>v>1.01&&v<85).sort((a,b)=>a[1]-b[1]);
  if(arr.length<2)return null;
  const[f1,o1]=arr[0],[f2,o2]=arr[1],i1=BAREMS_NBA.indexOf(f1),i2=BAREMS_NBA.indexOf(f2);
  const f2Up=i2>i1,req=(MID_NBA[f1]||205.5)-curTot;
  const pas=r=>({play:false,reason:r,f1,o1,f2,o2,req});
  if(gap<6)return pas("NBA 2K27 skor farkı çok yakın ("+gap+" < 6)");
  if(o1<1.20)return pas("NBA 2K27 oran çok düşük ("+o1+")");
  if(o1>=1.20&&o1<1.50)return(gap<=20)?{play:true,b:f1,o:o1,f:1,req}:pas("NBA 2K27 1.20-1.50 fark > 20");
  if(o1>=1.50&&o1<1.85){
    return(req<=49.5||!f2Up)?{play:true,b:f2,o:o2,f:2,req}:pas("NBA 2K27 1.50-1.85 filtresi");
  }
  if(o1>=1.85&&o1<2.10){
    if(gap<=15&&!f2Up&&req<=52.5)return{play:true,b:f1,o:o1,f:1,req};
    if((!f2Up||gap>=16||ps!==1)&&o2<=4.20)return{play:true,b:f2,o:o2,f:2,req};
    return pas("NBA 2K27 1.85-2.10 beklenti dışı");
  }
  if(o1>=2.10&&o1<2.30){
    return(gap>=16||req<=49.5||ps===1||f2Up)?{play:true,b:f1,o:o1,f:1,req}:pas("NBA 2K27 2.10-2.30 filtresi");
  }
  if(o1>=2.30&&o1<2.50){
    if(req<=49.5||ps===1||(f2Up&&gap<=15))return{play:true,b:f1,o:o1,f:1,req};
    if(!f2Up&&gap<=15&&o2<=3.40)return{play:true,b:f2,o:o2,f:2,req};
    return pas("NBA 2K27 2.30-2.50 beklenti dışı");
  }
  if(o1>=2.50)return(o2<=4.50)?{play:true,b:f2,o:o2,f:2,req}:pas("NBA 2K27 2.50+ oran yüksek");
  return pas("Kural dışı oran yapısı");
}

function ruleEuro(om,gap,ps,curTot){
  const arr=Object.entries(om).filter(([,v])=>v>1.01&&v<85).sort((a,b)=>a[1]-b[1]);
  if(arr.length<2)return null;
  const[f1,o1]=arr[0],[f2,o2]=arr[1],i1=BAREMS_EU.indexOf(f1),i2=BAREMS_EU.indexOf(f2);
  const om1=i1>0?(om[BAREMS_EU[i1-1]]||99):99,op1=(i1>=0&&i1<5)?(om[BAREMS_EU[i1+1]]||99):99,up=op1<om1,f2Up=i2>i1;
  const req=(MID_EU[f1]||155.5)-curTot,odiff=o2-o1;
  const pas=r=>({play:false,reason:r,f1,o1,f2,o2,req});
  if(gap<6)return pas("Skor farkı çok yakın ("+gap+" sayı < 6, uzatma riski)");
  if(gap>=29)return pas("Aşırı fark ("+gap+" >= 29 sayı, rölanti riski)");
  if(o1<1.20||o1>=2.30)return pas("Euro oran aralığı dışı (1.Fav: "+o1+")");
  if(o1>=1.20&&o1<1.50){
    return(gap<=15&&f2Up&&req<=38.5)?{play:true,b:f1,o:o1,f:1,req}:pas("Euro 1.20-1.50 filtresi (Fark:"+gap+", İstenen:"+req.toFixed(1)+")");
  }
  if(o1>=1.50&&o1<1.85){
    if(f2Up&&req<=37.5)return{play:true,b:f1,o:o1,f:1,req};
    if(gap>=21&&req<=41.5)return{play:true,b:f1,o:o1,f:1,req};
    return pas("Euro 1.50-1.85 alt barem riski (Fark:"+gap+", İstenen:"+req.toFixed(1)+")");
  }
  if(o1>=1.85&&o1<2.10){
    if(gap>=16&&gap<=28&&!f2Up&&req>=37.5&&o2<=3.40)return{play:true,b:f2,o:o2,f:2,req};
    if(gap<=15&&((f2Up&&req<=39.5)||(odiff<=0.18&&req<=37.5)))return{play:true,b:f1,o:o1,f:1,req};
    return pas("Euro 1.85-2.10 yön/tempo filtresi (1.Fav İstenen:"+req.toFixed(1)+", Fark:"+gap+")");
  }
  if(o1>=2.10&&o1<2.30){
    if(req<=36.5&&(f2Up||ps===1||gap>=16))return{play:true,b:f1,o:o1,f:1,req};
    if(req>36.5&&req<=41.5&&o2<=3.40&&gap<=24)return{play:true,b:f2,o:o2,f:2,req};
    return pas("Euro 2.10-2.30 beklenti dışı (1.Fav İstenen:"+req.toFixed(1)+", 2.Fav:"+o2+")");
  }
  return pas("Kural dışı oran yapısı");
}

async function checkResults(){
  const op=gOpen();
  if(!op.length)return;
  try{
    const wTo=Math.floor(Date.now()/300000)*300+300,wFrom=wTo-86400;
    const resMap={};
    for(const lg of[{k:"NBA",c:"2935701",minT:100},{k:"NBA27",c:"3125701",minT:100},{k:"EURO",c:"2983166",minT:60}]){
      const url=location.origin+"/service-api/result/web/api/v3/games?champId="+lg.c+"&dateFrom="+wFrom+"&dateTo="+wTo+"&lng=en&ref=110";
      const r=await fetch(url,{cache:"no-store"});
      if(!r.ok)continue;
      const items=(await r.json()).items||[];
      for(const it of items){
        const m=String(it.score||"").match(/^(\d+):(\d+)/);
        if(it.id&&m){
          const h=parseInt(m[1],10),a=parseInt(m[2],10);
          if(h!==a&&(h+a)>=lg.minT)resMap[String(it.id)]={h,a,tot:h+a,b:scoreToBarem(h+a,lg.k)};
        }
      }
    }
    const rem=[];
    let settled=false;
    for(const b of op){
      const rs=resMap[String(b.gid)];
      if(!rs){if(Date.now()-b.ts<43200000)rem.push(b);continue}
      settled=true;
      const won=(rs.b===b.barem);
      const net=won?+(b.stake*(b.odd-1)).toFixed(2):-b.stake;
      const st=gStats();
      if(won)st.w+=1;else st.l+=1;
      st.net=+(st.net+net).toFixed(2);
      st.hist=(st.hist||[]).slice(-19);
      st.hist.push((won?"✅":"❌")+" ["+b.lg+"] "+b.barem+" @"+b.odd+" ("+(net>=0?"+":"")+net.toFixed(1)+"₺)");
      jSet("n78_stats",st);
      await sleep(3500);
      const w=readWallet(),tr=getTier1500(w.num);
      let kMsg="";
      if(w.num&&tr.s1!==gS1())kMsg="\n💡 <b>Kademe Bilgisi:</b> Bakiyen Kademe "+tr.k+" aralığında (<code>/tutar "+tr.s1+" "+tr.s2+"</code> yazarak güncelleyebilirsin).";
      await sendTg(
        (won?"✅ <b>KUPON KAZANDI! (+"+net.toFixed(1)+" TL)</b>":"❌ <b>KUPON KAYBETTİ ("+net.toFixed(1)+" TL)</b>")+"\n\n"+
        "🏀 <b>["+b.lg+"] "+b.teams+"</b>\n"+
        "🏁 Maç Sonu: <code>"+rs.h+"-"+rs.a+"</code> (Toplam: <b>"+rs.tot+"</b> ➔ <code>"+rs.b+"</code>)\n"+
        "🎯 Oynanan: "+b.favType+" <b>"+b.barem+"</b> @ <b>"+b.odd+"</b> ("+b.stake+" TL)\n\n"+
        "📊 <b>Toplam:</b> "+st.w+" Kazandı / "+st.l+" Kaybetti (Net: <b>"+(st.net>=0?"+":"")+st.net.toFixed(1)+" TL</b>)\n"+
        "💰 <b>Güncel Melbet Bakiye:</b> "+w.txt+kMsg
      );
    }
    if(settled){jSet("n78_open",rem);ui()}
  }catch{}
}

async function pollTg(){
  if(TG_TOK.includes("__TG_"))return;
  const lockTs=Number(localStorage.getItem("n78_tglk")||"0");
  if(Date.now()-lockTs<5000)return;
  localStorage.setItem("n78_tglk",String(Date.now()));
  try{
    const off=Number(localStorage.getItem("n78_upd")||"0");
    const r=await fetch("https://api.telegram.org/bot"+TG_TOK+"/getUpdates?offset="+(off+1)+"&timeout=0",{cache:"no-store"});
    if(!r.ok)return;
    const arr=(await r.json()).result||[];
    for(const u of arr){
      localStorage.setItem("n78_upd",String(u.update_id));
      const m=u.message;
      if(!m||!m.chat)continue;
      const cid=String(m.chat.id);
      localStorage.setItem("n78_cid",cid);
      const txt=String(m.text||"").trim(),low=txt.toLowerCase();
      if(low.startsWith("/start")||low==="merhaba"||low.startsWith("/kasa")||low.startsWith("/durum")){
        await sendTg(buildStatusMsg(),cid);
      }else if(low.startsWith("/tutar")){
        const p=txt.split(/\s+/),v1=parseFloat(p[1]),v2=parseFloat(p[2]);
        if(v1>0&&v2>0){
          localStorage.setItem("n77_s1",String(v1));localStorage.setItem("n77_s2",String(v2));ui();
          await sendTg("✅ <b>Bahis Tutarları Güncellendi!</b>\n1. Favori: <b>"+v1+" TL</b>\n2. Favori: <b>"+v2+" TL</b>",cid);
        }else await sendTg("⚠️ Örnek kullanım: <code>/tutar 105 85</code>",cid);
      }else if(low.startsWith("/durdur")){
        localStorage.setItem("n77_on","0");ui();
        await sendTg("🔴 <b>Otomatik Bahis Durduruldu.</b> (/baslat ile açabilirsin)",cid);
      }else if(low.startsWith("/baslat")){
        localStorage.setItem("n77_on","1");ui();
        await sendTg("🟢 <b>7/24 Otomatik Bahis Aktif!</b>",cid);
      }else if(low.startsWith("/sifirla")){
        jSet("n78_stats",{w:0,l:0,net:0,hist:[]});ui();
        await sendTg("🔄 <b>Kazanç / kayıp sayacı sıfırlandı.</b>",cid);
      }else if(low.startsWith("/pas")){
        const nx=isPasTgOn()?"0":"1";
        localStorage.setItem("n80_pastg",nx);
        await sendTg(nx==="1"?"🟢 <b>PAS geçilen maç bildirimleri AÇILDI.</b>":"🔴 <b>PAS geçilen maç bildirimleri KAPATILDI.</b>",cid);
      }
    }
  }catch{}
}

async function syncC(){try{const r=await fetch(CLOUD+"?t="+Date.now(),{cache:"no-store"});if(r.ok){const d=await r.json();if(d.chain_state)pShiftNba=Number(d.chain_state.prev_shift||0);if(d.euro_chain_state)pShiftEu=Number(d.euro_chain_state.prev_shift||0)}}catch{}}

async function poll(){
  if(busy)return;
  ui();
  try{
    let totG=0,msg="";
    const leagues=[
      {k:"NBA",c:"2935701",s:"2935701-nba-2k26-cyber-league",qSec:720,qMin:12,minLo:34.2,minHi:35.8},
      {k:"NBA27",c:"3125701",s:"3125701-nba-2k27-cyber-league",qSec:720,qMin:12,minLo:34.2,minHi:35.8},
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
        const teams=String(g.O1||"").replace(" (cyber)","")+" vs "+String(g.O2||"").replace(" (cyber)","");
        if(el>=lg.minLo&&el<=lg.minHi){
          const r2=await fetch(location.origin+"/service-api/LiveFeed/GetGameZip?id="+gid+"&lng=tr&isSubGames=true&GroupEvents=true",{cache:"no-store"});
          if(!r2.ok)continue;
          const g33=((await r2.json()).Value?.GE||[]).find(x=>x.G===3319);
          if(!g33)continue;
          const om={};
          for(const rw of g33.E||[])for(const ev of rw){
            if(ev.B||ev.L)continue;
            const bk=lg.k==="EURO"?PMAP_EU[String(ev.P||"")]:PMAP_NBA[String(ev.P||"")];
            if(bk&&ev.C>1.01)om[bk]=Number(ev.C);
          }
          const dec=lg.k==="NBA"?ruleNba(om,gap,pShiftNba,curTot):(lg.k==="NBA27"?ruleNba27(om,gap,pShift27,curTot):ruleEuro(om,gap,pShiftEu,curTot));
          const sid="CB-"+gid+"_"+lg.k,pl=gP();
          if(dec&&dec.play){
            const stTL=dec.f===1?gS1():gS2();
            msg='<b style="color:#34d399">🚨 ['+lg.k+'] '+el.toFixed(1)+'.DK OYNA!</b><br>🎯 '+dec.b+' @ '+dec.o+' ('+stTL+' TL)';
            if(isOn()&&!pl.includes(sid)&&pl.length<gMax()){
              const pObj={gid,sid,lg:lg.k,b:dec.b,o:dec.o,f:dec.f,stTL,teams,sc:s1+"-"+s2,ts:Date.now()};
              if(!location.href.includes(gid)){
                sessionStorage.setItem("n77_pend",JSON.stringify(pObj));
                location.href=location.origin+"/tr/live/basketball/"+lg.s+"/"+gid;
                return;
              }
              await sleep(1200);
              const cr=await placeMelbetBet(dec.b,Math.max(1.2,dec.o-0.18),stTL);
              if(cr.ok)await recordAndNotifyBet(pObj,cr);
            }
          }else if(dec&&!dec.play){
            msg='⛔ ['+lg.k+'] '+el.toFixed(1)+'.dk PAS ('+s1+'-'+s2+')';
            const pasMin=lg.k==="EURO"?28.75:34.75;
            if(el>=pasMin&&!pl.includes(sid)&&!gPas().includes(sid)){
              mPas(sid);
              if(isPasTgOn()){
                const normTxt=lg.k==="EURO"?"~38.9":"~54.1";
                await sendTg(
                  "⛔ <b>MAÇ PAS GEÇİLDİ ["+lg.k+"]</b>\n\n"+
                  "🏀 <b>"+teams+"</b>\n"+
                  "⏱️ Dakika: <b>"+el.toFixed(1)+".dk</b> | Skor: <code>"+s1+"-"+s2+"</code> (Toplam: <b>"+curTot+"</b>, Fark: <b>"+gap+"</b>)\n"+
                  "📊 1.Fav: <code>"+dec.f1+"</code> (@<b>"+dec.o1+"</b>) | 2.Fav: <code>"+dec.f2+"</code> (@<b>"+dec.o2+"</b>)\n"+
                  "🎯 1.Fav Baremin İstediği Kalan Sayı: <b>"+dec.req.toFixed(1)+" sayı</b> (Normali: "+normTxt+")\n"+
                  "🛡️ <b>Pas Sebebi:</b> "+dec.reason
                );
              }
            }
          }
        }
      }
    }
    if(!msg)msg='📡 NBA + EURO v8.1 • <b>'+totG+' maç</b>';
    E("n77s").innerHTML=msg+'<div style="font-size:10px;color:#94a3b8">'+new Date().toLocaleTimeString("tr-TR")+' • NBA:'+pShiftNba+' EU:'+pShiftEu+' • Bekleyen:'+gOpen().length+'</div>';
  }catch{}
}
syncC();
setInterval(syncC,30000);
setInterval(poll,5000);
setInterval(pollTg,6000);
setInterval(checkResults,15000);
checkPendingAutoTest().then(()=>{poll();pollTg();checkResults()});
})();
