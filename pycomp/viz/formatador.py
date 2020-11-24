"""
---------------------------------------------------
------------ TÓPICO: Viz - Formatador -------------
---------------------------------------------------
Módulo responsável por definir funções de formatação
de gráficos e rótulos construídos a partir das 
ferramentas de plotagem matplotlib e seaborn

Sumário
-----------------------------------

-----------------------------------
"""

# Importando bibliotecas
import matplotlib
from matplotlib.patches import Patch
from matplotlib.axes import Axes

# AnnotateBars class (referência na classe)
from dataclasses import dataclass
from typing import *


"""
---------------------------------------------------
--------- 1. FORMATAÇÃO DE EIXOS E RÓTULOS --------
                1.1 Eixos de plotagens
---------------------------------------------------
"""

# Formatando eixos do matplotlib
def format_spines(ax, right_border=True):
    """
    Função responsável por modificar as bordas e cores de eixos do matplotlib

    Parâmetros
    ----------
    :param ax: eixo do gráfico criado no matplotlib [type: matplotlib.pyplot.axes]
    :param right_border: flag para plotagem ou ocultação da borda direita [type: bool, default=True]

    Retorno
    -------
    Esta função não retorna nenhum parâmetro além do eixo devidamente customizado

    Aplicação
    ---------
    fig, ax = plt.subplots()
    format_spines(ax=ax, right_border=False)
    """

    # Definindo cores dos eixos
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['top'].set_visible(False)

    # Validando plotagem da borda direita
    if right_border:
        ax.spines['right'].set_color('#CCCCCC')
    else:
        ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')

"""
---------------------------------------------------
--------- 1. FORMATAÇÃO DE EIXOS E RÓTULOS --------
        1.2 Criação e formatação de rótulos
---------------------------------------------------
"""

# Referência: https://towardsdatascience.com/annotating-bar-charts-and-other-matplolib-techniques-cecb54315015
# Criando allias
#Patch = matplotlib.patches.Patch
PosVal = Tuple[float, Tuple[float, float]]
#Axis = matplotlib.axes.Axes
Axis = Axes
PosValFunc = Callable[[Patch], PosVal]

@dataclass
class AnnotateBars:
    font_size: int = 10
    color: str = "black"
    n_dec: int = 2
    def horizontal(self, ax: Axis, centered=False):
        def get_vals(p: Patch) -> PosVal:
            value = p.get_width()
            div = 2 if centered else 1
            pos = (
                p.get_x() + p.get_width() / div,
                p.get_y() + p.get_height() / 2,
            )
            return value, pos
        ha = "center" if centered else  "left"
        self._annotate(ax, get_vals, ha=ha, va="center")
    def vertical(self, ax: Axis, centered:bool=False):
        def get_vals(p: Patch) -> PosVal:
            value = p.get_height()
            div = 2 if centered else 1
            pos = (p.get_x() + p.get_width() / 2,
                   p.get_y() + p.get_height() / div
            )
            return value, pos
        va = "center" if centered else "bottom"
        self._annotate(ax, get_vals, ha="center", va=va)
    def _annotate(self, ax, func: PosValFunc, **kwargs):
        cfg = {"color": self.color,
               "fontsize": self.font_size, **kwargs}
        for p in ax.patches:
            value, pos = func(p)
            ax.annotate(f"{value:.{self.n_dec}f}", pos, **cfg)


# Definindo funções úteis para plotagem dos rótulos no gráfico
def make_autopct(values):
    """
    Função para configuração de rótulos em gráficos de rosca

    Parâmetros
    ----------
    :param values: valores atrelados ao rótulo [type: np.array]

    Retorno
    -------
    :return my_autopct: string formatada para plotagem dos rótulos
    """

    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))

        return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)

    return my_autopct