import os, json, time, uuid, base64, subprocess, traceback, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Optional
import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(os.getenv('OUTPUT_DIR', '/data')).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
API_KEY = os.getenv('STUDIO_API_KEY', '')
OPENAI_KEY = os.getenv('OPENAI_API_KEY', '')
ELEVEN_KEY = os.getenv('ELEVENLABS_API_KEY', '')
VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID', '')
TEXT_MODEL = os.getenv('TEXT_MODEL', 'gpt-4.1-mini')
IMAGE_MODEL = os.getenv('IMAGE_MODEL', 'gpt-image-1-mini')
POOL = ThreadPoolExecutor(max_workers=1)
LOCK = Lock()
app = FastAPI(title='Mistério-SOS Studio', version='1.0.0')

class Pedido(BaseModel):
    tema: str = Field(min_length=6, max_length=350)
    tipo: str = Field(default='real', pattern='^(real|ficcao)$')
    cenas: int = Field(default=8, ge=3, le=16)
    fatos_confirmados: str = Field(default='', max_length=12000)
    estilo: str = Field(default='cinematografico realista', max_length=250)


def auth(x_studio_key: Optional[str]):
    if not API_KEY or x_studio_key != API_KEY:
        raise HTTPException(status_code=401, detail='Chave de acesso inválida')


def save_status(job, status, etapa, erro=None):
    with LOCK:
        file = ROOT / job / 'status.json'
        state = json.loads(file.read_text('utf-8')) if file.exists() else {'job_id':job}
        state.update({'status':status,'etapa':etapa,'atualizado_em':int(time.time())})
        if erro: state['erro'] = str(erro)[-800:]
        file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def call_json(url, payload, headers, timeout=180):
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def roteiro(req):
    prompt = f'''Escreva em português do Brasil, com narrativa original e envolvente para o canal Arquivos Mistério-SOS. Tema: {req.tema}. Tipo: {req.tipo}. Crie EXATAMENTE {req.cenas} cenas sequenciais. Cada cena deve ter 35 a 75 palavras narradas, e um prompt visual em INGLÊS para imagem 16:9 realista, não para vídeo. Não coloque números de cena, rubricas ou títulos no texto da locução. Mantenha navio/personagens/ambiente consistentes entre cenas. Evite texto legível dentro das imagens. Devolva APENAS um objeto JSON válido com chaves titulo, aviso, cenas; cenas é array de objetos {{"narracao":"...","prompt_imagem":"..."}}. Para histórias REAIS, use apenas os fatos confirmados fornecidos abaixo. Não invente datas, falas, cenas documentais ou detalhes específicos como comprovados; quando os dados forem insuficientes, use formulações cautelosas e o aviso informe que há reconstruções visuais. Para histórias FICTÍCIAS, indique explicitamente no aviso que são ficção. Fatos fornecidos: {req.fatos_confirmados or '(nenhum; tratar tudo como hipótese ou descrição genérica, sem inventar fatos)'}'''
    data = call_json('https://api.openai.com/v1/chat/completions', {
        'model':TEXT_MODEL, 'messages':[{'role':'system','content':'Você devolve somente JSON válido conforme solicitado.'},{'role':'user','content':prompt}],
        'response_format':{'type':'json_object'}
    }, {'Authorization':f'Bearer {OPENAI_KEY}'}, 180)
    content = json.loads(data['choices'][0]['message']['content'])
    cenas = content.get('cenas',[])
    if len(cenas) != req.cenas or any(not s.get('narracao') or not s.get('prompt_imagem') for s in cenas):
        raise ValueError('IA não retornou exatamente o número de cenas solicitado; tente novamente')
    return content


def image_generate(prompt, target):
    data = call_json('https://api.openai.com/v1/images/generations', {
        'model':IMAGE_MODEL, 'prompt':prompt, 'size':'1536x1024', 'quality':'low', 'n':1
    }, {'Authorization':f'Bearer {OPENAI_KEY}'}, 240)
    entry = data['data'][0]
    if entry.get('b64_json'):
        target.write_bytes(base64.b64decode(entry['b64_json']))
    elif entry.get('url'):
        r=requests.get(entry['url'],timeout=120);r.raise_for_status();target.write_bytes(r.content)
    else: raise ValueError('API de imagem não devolveu imagem')


def speech_generate(text, target):
    url = f'https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format=mp3_44100_128'
    payload={'text':text, 'model_id':'eleven_multilingual_v2',
             'voice_settings':{'stability':0.55,'similarity_boost':0.8,'style':0.1,'use_speaker_boost':True}}
    r=requests.post(url,json=payload,headers={'xi-api-key':ELEVEN_KEY,'Accept':'audio/mpeg'},timeout=180)
    r.raise_for_status();target.write_bytes(r.content)


def duration(path):
    p=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)],capture_output=True,text=True,check=True)
    return float(p.stdout.strip())


def run_command(cmd):
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode: raise RuntimeError('FFmpeg: '+p.stderr[-1600:])


def render_scene(image,audio,out,idx):
    seconds=duration(audio)+0.28
    fps=24
    frames=round(seconds*fps)
    # Render image from a high-resolution intermediary; center-lock pan eliminates horizontal jitter.
    z=(f'1.04-0.04*on/{max(frames-1,1)}' if idx % 2 == 0
       else f'1+0.04*on/{max(frames-1,1)}')
    vf=f'scale=2560:1440:force_original_aspect_ratio=increase,crop=2560:1440,zoompan=z=\'{z}\':x=\'iw/2-(iw/zoom/2)\':y=\'ih/2-(ih/zoom/2)\':d={frames}:s=1280x720:fps=24,format=yuv420p'
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-i',str(image),'-i',str(audio),'-vf',vf,
         '-af','apad','-t',f'{seconds:.3f}','-r','24','-c:v','libx264','-preset','veryfast','-crf','21',
         '-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-ar','44100','-ac','2','-movflags','+faststart',str(out)]
    run_command(cmd)
    return seconds


def process(job,req):
    folder=ROOT/job
    try:
        save_status(job,'processando','Gerando roteiro')
        script=roteiro(req)
        (folder/'roteiro.json').write_text(json.dumps(script,ensure_ascii=False,indent=2),encoding='utf-8')
        scenes=[]
        total=0
        for i,scene in enumerate(script['cenas'],1):
            save_status(job,'processando',f'Cena {i}/{req.cenas} — gerando imagem e narração')
            img=folder/f'cena_{i:02d}.png';audio=folder/f'cena_{i:02d}.mp3'
            prefix=f'Cinematic historically plausible 16:9 film still, {req.estilo}, consistent cast and setting, no visible words, no watermarks. '
            image_generate(prefix+scene['prompt_imagem'],img)
            speech_generate(scene['narracao'],audio)
            save_status(job,'processando',f'Cena {i}/{req.cenas} — montando vídeo')
            output=folder/f'cena_{i:02d}.mp4'
            sec=render_scene(img,audio,output,i)
            scenes.append(output)
            total+=sec
        # All scenes share encoder, size, FPS and audio sampling.
        listfile=folder/'partes.txt'
        listfile.write_text(''.join(f"file '{s.name}'\n" for s in scenes),encoding='utf-8')
        final=folder/'video_final.mp4'
        run_command(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listfile),'-c','copy','-movflags','+faststart',str(final)])
        save_status(job,'concluido',f'Vídeo gerado ({duration(final):.1f} segundos)')
    except Exception as e:
        (folder/'erro_detalhado.txt').write_text(traceback.format_exc(),encoding='utf-8')
        save_status(job,'erro','Falha na produção',e)


@app.get('/health')
def health(): return {'ok':True,'service':'misterio-sos-studio'}

@app.post('/jobs',status_code=202)
def create_job(pedido:Pedido,x_studio_key:Optional[str]=Header(default=None)):
    auth(x_studio_key)
    if not (OPENAI_KEY and ELEVEN_KEY and VOICE_ID):
        raise HTTPException(status_code=503,detail='Configure OPENAI_API_KEY, ELEVENLABS_API_KEY e ELEVENLABS_VOICE_ID')
    job=uuid.uuid4().hex[:16]
    (ROOT/job).mkdir()
    save_status(job,'fila','Aguardando início')
    POOL.submit(process,job,pedido)
    return {'job_id':job,'status':'fila','consulta':f'/jobs/{job}', 'video':f'/jobs/{job}/video'}

@app.get('/jobs/{job}')
def get_job(job:str,x_studio_key:Optional[str]=Header(default=None)):
    auth(x_studio_key)
    if not re.fullmatch('[a-f0-9]{16}',job): raise HTTPException(404,'Não encontrado')
    status=ROOT/job/'status.json'
    if not status.exists(): raise HTTPException(404,'Não encontrado')
    return json.loads(status.read_text('utf-8'))

@app.get('/jobs/{job}/video')
def download(job:str,x_studio_key:Optional[str]=Header(default=None)):
    auth(x_studio_key)
    if not re.fullmatch('[a-f0-9]{16}',job): raise HTTPException(404,'Não encontrado')
    target=ROOT/job/'video_final.mp4'
    if not target.exists(): raise HTTPException(404,'Vídeo ainda não disponível')
    return FileResponse(target,media_type='video/mp4',filename=f'misterio_sos_{job}.mp4')
