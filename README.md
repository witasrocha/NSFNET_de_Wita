# Requisitos

- CUDA 10 e CUDnn 7.3
- Uso do Tensorflow 1.14
- Tensorflow-GPU 1.14
- Python 3.7
# Observações

- Utilizar a versão miniconda com python 3.6 (versão 4.3.31)
- Utilizar uma IDE (Pycharm compatível)
- Aplicar a versão 1.14 do tensorflow por enquanto

# NSFnets
"Rede PINN" com proposta de resolver as equações de Navier–Stokes

## Diferenças entre os arquivos disponíveis

VP_NSFnets1.py simula o fluxo Kovasznay (2 dimensões no estado estacionário)  
VP_NSFnets2.py simula o cilindro (2 dimensões e transiente)  
VP_NSFnets3.py simula o fluxo Beltrami(3 dimensões e transiente)  
VP_NSFnets4.py is simula o fluxo em canal turbulento(3 dimensões e transiente)  


## Função de perda

$$\begin{aligned} L &=L_{e}+\alpha L_{b}+\beta L_{i} \\ L_{e} &=\frac{1}{N_{e}} \sum_{i=1}^{4} \sum_{n=1}^{N_{e}}\left|e_{V P i}^{n}\right|^{2} \\ L_{b} &=\frac{1}{N_{b}} \sum_{n=1}^{N_{b}}\left|\mathbf{u}^{n}-\mathbf{u}_{b}^{n}\right|^{2} \\ L_{i} &=\frac{1}{N_{i}} \sum_{n=1}^{N_{i}}\left|\mathbf{u}^{n}-\mathbf{u}_{i}^{n}\right|^{2} \end{aligned}$$

$$\begin{aligned}
u(x, y, z, t)=&-a\left[e^{a x} \sin (a y+d z)+e^{a z} \cos (a x+d y)\right] e^{-d^{2} t} \\
v(x, y, z, t)=&-a\left[e^{a y} \sin (a z+d x)+e^{a x} \cos (a y+d z)\right] e^{-d^{2} t} \\
w(x, y, z, t)=&-a\left[e^{a z} \sin (a x+d y)+e^{a y} \cos (a z+d x)\right] e^{-d^{2} t} \\
p(x, y, z, t)=&-\frac{1}{2} a^{2}\left[e^{2 a x}+e^{2 a y}+e^{2 a z}+2 \sin (a x+d y) \cos (a z+d x) e^{a(y+z)}\right.\\
&+2 \sin (a y+d z) \cos (a x+d y) e^{a(z+x)} \\
&\left.+2 \sin (a z+d x) \cos (a y+d z) e^{a(x+y)}\right] e^{-2 d^{2} t}
\end{aligned}$$

$$ e_{V P 1}= \partial_{t} u+u \partial_{x} u+v \partial_{y} u+w \partial_{z} u+\partial_{x} p-1 / Re\left(\partial_{x x}^{2} u+\partial_{y y}^{2} u+\partial_{z z}^{2} u\right) $$

## Solução analítica do fluxo Beltrami

$$\begin{aligned}
u(x, y, z, t)=&-a\left[e^{a x} \sin (a y+d z)+e^{a z} \cos (a x+d y)\right] e^{-d^{2} t} \\
v(x, y, z, t)=&-a\left[e^{a y} \sin (a z+d x)+e^{a x} \cos (a y+d z)\right] e^{-d^{2} t} \\
w(x, y, z, t)=&-a\left[e^{a z} \sin (a x+d y)+e^{a y} \cos (a z+d x)\right] e^{-d^{2} t} \\
p(x, y, z, t)=&-\frac{1}{2} a^{2}\left[e^{2 a x}+e^{2 a y}+e^{2 a z}+2 \sin (a x+d y) \cos (a z+d x) e^{a(y+z)}\right.\\
&+2 \sin (a y+d z) \cos (a x+d y) e^{a(z+x)} \\
&\left.+2 \sin (a z+d x) \cos (a y+d z) e^{a(x+y)}\right] e^{-2 d^{2} t}
\end{aligned}$$
