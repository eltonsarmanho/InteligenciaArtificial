import json
d=json.load(open('/home/claude/aula8_llm/llm_dados_final.json'))
pc=lambda x:f"{100*x:.1f}".replace('.',',')+r"\%"
def tk(t): return t.replace(' ','\\textvisiblespace{}').replace('_','\\_')
M=d['modelo']; top=d['p1']['top'][:6]; T=d['temp']; tok=d['tok']; am=d['amostras']; ex=d['amostras_exemplos']
q=lambda s:s.replace('"',"''").replace('“',"``").replace('”',"''").replace('\n',' ').replace('...','\\dots{}').replace('#','\\#')
al=d['alucinacao']['raw']['passos'][1]['top'][:8]
cot=d['cot']['problemas'][d['cot']['principal']]
L=[f"\\newcommand{{\\barsBel}}{{{' '.join(f'({i},{100*t["p"]:.1f})' for i,t in enumerate(top))}}}",
 "\\newcommand{\\labsBel}{"+",".join("{\\texttt{"+tk(t['tok'])+"}}" for t in top)+"}",
 *[f"\\newcommand{{\\bars{n}}}{{{' '.join(f'({i},{100*t["p"]:.1f})' for i,t in enumerate(T[k][:4]))}}}" for n,k in (('Ta','0.5'),('Tb','1.0'),('Tc','1.5'))],
 "\\newcommand{\\labsT}{"+",".join("{\\texttt{"+tk(t['tok'])+"}}" for t in T['1.0'][:4])+"}",
 f"\\newcommand{{\\barsAl}}{{{' '.join(f'({i},{100*p:.1f})' for i,(t,p) in enumerate(al))}}}",
 "\\newcommand{\\labsAl}{"+",".join("{\\texttt{"+t+"}}" for t,p in al)+"}",
 *[f"\\newcommand{{\\tok{c}}}{{{' '.join(chr(92)+'tk{'+tk(x)+'}' for x in tok[i]['tokens'])}}}" for i,c in enumerate('ABCDEFGHI')],
 *[f"\\newcommand{{\\ntok{c}}}{{{len(tok[i]['tokens'])}}}" for i,c in enumerate('ABCDEFGHI')]]
V=dict(BELP=pc(top[0]['p']),AP=pc(top[1]['p']),BELd=f"{top[0]['p']:.3f}",Ad=f"{top[1]['p']:.3f}",Od=f"{top[2]['p']:.3f}",
 Tmeio=pc(T['0.5'][0]['p']),Tum=pc(T['1.0'][0]['p']),Tumemeio=pc(T['1.5'][0]['p']),
 NUC=", ".join("\\texttt{"+tk(x)+"}" for x in d['top_p']['nucleo']),NUCM=pc(d['top_p']['massa']),Mvinte=pc(d['top_p']['massa_top20']),
 Dzero=am['0.0']['distintas'],Dsete=am['0.7']['distintas'],Dquinze=am['1.5']['distintas'],
 Azero=q(am['0.0']['respostas'][0]),Asetea=q(am['0.7']['respostas'][ex['0.7'][0]]),Aseteb=q(am['0.7']['respostas'][ex['0.7'][1]]),
 Aquinzea=q(am['1.5']['respostas'][ex['1.5'][0]]),Aquinzeb=q(am['1.5']['respostas'][ex['1.5'][1]]),
 GREEDY=q(d['p1_greedy']['resposta'].strip()),Pseis=pc(d['alucinacao']['raw']['passos'][1]['p']),
 CTX=f"{M['llama.context_length']:,}".replace(',','.'),EMB=f"{M['llama.embedding_length']:,}".replace(',','.'),LAY=M['llama.block_count'],
 HEADS=M['llama.attention.head_count'],KV=M['llama.attention.head_count_kv'],FFN=f"{M['llama.feed_forward_length']:,}".replace(',','.'),
 VOC=f"{d['vocab']:,}".replace(',','.'),SIZE=M['general.size_label'],
 COTdireto=cot['direto'],COTcerto=cot['certo'],COTokd=d['cot']['acertos_direto'],COTokp=d['cot']['acertos_passo'],COTn=len(d['cot']['problemas']),
 FSa=d['few_shot']['few_shot'][0],FSb=d['few_shot']['few_shot'][1],FSc=d['few_shot']['few_shot'][2])
L+=[f"\\newcommand{{\\v{k}}}{{{v}}}" for k,v in V.items()]
open('gen_vals.tex','w').write('\n'.join(L)+'\n'); print(V)
