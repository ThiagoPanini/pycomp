"""
---------------------------------------------------
------------ TÓPICO: Viz - Insights -------------
---------------------------------------------------
Módulo responsável por definir funções prontas para
uma rica aplicação do processo de EDA (Exploratory
Data Analysis) em uma base de dados. O objetivo é
proporcionar análises gráficas complexas com poucas
linhas de código

Sumário
-----------------------------------

-----------------------------------
"""

# Importando bibliotecas
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from pycomp.viz.formatador import make_autopct, format_spines, AnnotateBars


"""
---------------------------------------------------
-------- 1. ANÁLISES GRÁFICAS CUSTOMIZADAS --------
---------------------------------------------------
"""

def save_fig(fig, output_path, img_name, tight_layout=True, dpi=300):
        """
        Método responsável por salvar imagens geradas pelo matplotlib/seaborn

        Parâmetros
        ----------
        :param fig: figura criada pelo matplotlib para a plotagem gráfica [type: plt.figure]
        :param output_file: caminho final a ser salvo (+ nome do arquivo em formato png) [type: string]
        :param tight_layout: flag que define o acerto da imagem [type: bool, default=True]
        :param dpi: resolução da imagem a ser salva [type: int, default=300]

        Retorno
        -------
        Este método não retorna nenhum parâmetro além do salvamento da imagem em diretório especificado

        Aplicação
        ---------
        fig, ax = plt.subplots()
        save_fig(fig, output_file='imagem.png')
        """

        # Verificando se diretório existe
        if not os.path.isdir(output_path):
            print(f'Diretório {output_path} inexistente. Criando diretório no local especificado')
            try:
                os.makedirs(output_path)
            except Exception as e:
                print(f'Erro ao tentar criar o diretório {output_path}. Exception lançada: {e}')
                return
        
        # Acertando layout da imagem
        if tight_layout:
            fig.tight_layout()
        
        try:
            output_file = os.path.join(output_path, img_name)
            fig.savefig(output_file, dpi=300)
        except Exception as e:
            print(f'Erro ao salvar imagem. Exception lançada: {e}')
            return

def plot_donut_chart(df, col, **kwargs):
    """
    Função responsável por plotar um gráfico de rosca customizado para uma determinada coluna da base
    
    Parâmetros
    ----------
    :param df: base de dados utilizada na plotagem [type: pd.DataFrame]
    :param col: nome da coluna a ser analisada [type: string]
    :param **kwargs: parâmetros adicionais da função
        :arg figsize: dimensões da figura de plotagem [type: tuple, default=(8, 8)]
        :arg circle_radius: raio do círculo central do gráfico [type: float, default=0.8]
        :arg circle_radius_color: cor do círculo central do gráfico [type: string, default='white']
        :arg label_names: labels personalizados para os rótulos [type: dict, default=value_counts().index]
        :arg top: índice de filtro das top categorias a serem plotadas [type: int]
        :arg colors: lista de cores para aplicação na plotagem [type: list]
        :arg text: texto central do gráfico de rosca [type: string, default=f'Total: \n{sum(values)}']
        :arg title: título do gráfico [type: string, default=f'Gráfico de Rosca para a Variável ${col}$']
        :arg autotexts_size: dimensão do rótulo do valor numérico do gráfico [type: int, default=14]
        :arg autotexts_color: cor do rótulo do valor numérico do gráfico [type: int, default='black]
        :arg texts_size: dimensão do rótulo do label [type: int, default=14]
        :arg texts_color: cor do rótulo do label [type: int, default='black']
        :arg save: flag indicativo de salvamento da imagem gerada [type: bool, default=None]
        :arg output_path: caminho de output da imagem a ser salva [type: string, default='output/']
        :arg img_name: nome do arquivo .png a ser gerado [type: string, default=f'{col}_donutchart.png']
    
    Retorno
    -------
    Essa função não retorna nenhum parâmetro além da plotagem customizada do gráfico de rosca

    Aplicação
    ---------
    plot_donut_chart(df=df, col='categorical_column', label_names={1: 'Classe 1', 2: 'Classe 2'})
    """
    
    # Validando presença da coluna na base
    if col not in df.columns:
        print(f'Coluna {col} não presente na base')
        return

    # Retornando vales e labels para plotagem
    counts = df[col].value_counts()
    values = counts.values
    labels = counts.index
    if 'label_names' in kwargs:
        try:
            labels = labels.map(kwargs['label_names'])
        except Exception as e:
            print(f'Erro ao mapear o dicionário label_names na coluna {col}. Exception: {e}')

    # Verificando filtro de top categorias na análise
    if 'top' in kwargs and kwargs['top'] > 0:
        values = values[:-kwargs['top']]
        labels = labels[:-kwargs['top']]
    
    # Cores para a plotagem
    color_list = ['darkslateblue', 'crimson', 'lightseagreen', 'lightskyblue', 'lightcoral', 'silver']
    colors = kwargs['colors'] if 'colors' in kwargs else color_list[:len(labels)]

    # Parâmetros de plotagem
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (8, 8)
    circle_radius = kwargs['circle_radius'] if 'circle_radius' in kwargs else 0.8
    circle_radius_color = kwargs['circle_radius_color'] if 'circle_radius_color' in kwargs else 'white'

    # Plotando gráfico de rosca
    center_circle = plt.Circle((0, 0), circle_radius, color=circle_radius_color)
    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, startangle=90, autopct=make_autopct(values))
    ax.add_artist(center_circle)

    # Configurando argumentos do texto central
    text = kwargs['text'] if 'text' in kwargs else f'Total: \n{sum(values)}'
    text_kwargs = dict(size=20, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **text_kwargs)
    
    # Definindo título
    title = kwargs['title'] if 'title' in kwargs else f'Gráfico de Rosca para a Variável ${col}$'
    ax.set_title(title, size=16, color='dimgrey')

    # Parâmetros de customização do gráfico gerado
    autotexts_size = kwargs['autotexts_size'] if 'autotexts_size' in kwargs else 14
    autotexts_color = kwargs['autotexts_color'] if 'autotexts_color' in kwargs else 'black'
    texts_size = kwargs['texts_size'] if 'texts_size' in kwargs else 14
    texts_color = kwargs['texts_color'] if 'texts_stexts_colorize' in kwargs else 'black'

    # Customizando rótulos
    plt.setp(autotexts, size=autotexts_size, color=autotexts_color)
    plt.setp(texts, size=texts_size, color=texts_color)

    # Verificando salvamento da imagem
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col}_donutchart.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

def plot_pie_chart(df, col, **kwargs):
    """
    Função responsável por plotar um gráfico de rosca customizado para uma determinada coluna da base
    
    Parâmetros
    ----------
    :param df: base de dados utilizada na plotagem [type: pd.DataFrame]
    :param col: nome da coluna a ser analisada [type: string]
    :param **kwargs: parâmetros adicionais da função
        :arg figsize: dimensões da figura de plotagem [type: tuple, default=(8, 8)]
        :arg label_names: labels personalizados para os rótulos [type: dict, default=value_counts().index]
        :arg top: índice de filtro das top categorias a serem plotadas [type: int]
        :arg colors: lista de cores para aplicação na plotagem [type: list]
        :arg explode: parâmetro para separação da fatia do gráfico [type: tuple, default=(0,)]
        :arg shadow: presença de sombra nas fatias do gráfico [type: bool, default=True]
        :arg title: título do gráfico [type: string, default=f'Gráfico de Rosca para a Variável ${col}$']
        :arg autotexts_size: dimensão do rótulo do valor numérico do gráfico [type: int, default=14]
        :arg autotexts_color: cor do rótulo do valor numérico do gráfico [type: int, default='black]
        :arg texts_size: dimensão do rótulo do label [type: int, default=14]
        :arg texts_color: cor do rótulo do label [type: int, default='black']
        :arg save: flag indicativo de salvamento da imagem gerada [type: bool, default=None]
        :arg output_path: caminho de output da imagem a ser salva [type: string, default='output/']
        :arg img_name: nome do arquivo .png a ser gerado [type: string, default=f'{col}_donutchart.png']
    
    Retorno
    -------
    Essa função não retorna nenhum parâmetro além da plotagem customizada do gráfico de pizza

    Aplicação
    ---------
    plot_pie_chart(df=df, col='categorical_column', label_names={1: 'Classe 1', 2: 'Classe 2'})
    """
    
    # Validando presença da coluna na base
    if col not in df.columns:
        print(f'Coluna {col} não presente na base')
        return

    # Retornando vales e labels para plotagem
    counts = df[col].value_counts()
    values = counts.values
    labels = counts.index
    if 'label_names' in kwargs:
        try:
            labels = labels.map(kwargs['label_names'])
        except Exception as e:
            print(f'Erro ao mapear o dicionário label_names na coluna {col}. Exception: {e}')

    # Verificando filtro de top categorias na análise
    if 'top' in kwargs and kwargs['top'] > 0:
        values = values[:-kwargs['top']]
        labels = labels[:-kwargs['top']]
    
    # Cores para a plotagem
    color_list = ['darkslateblue', 'crimson', 'lightseagreen', 'lightskyblue', 'lightcoral', 'silver']
    colors = kwargs['colors'] if 'colors' in kwargs else color_list[:len(labels)]

    # Parâmetros de plotagem
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (8, 8)
    explode = kwargs['explode'] if 'explode' in kwargs else (0,) * len(labels)
    shadow = kwargs['shadow'] if 'shadow' in kwargs else False
    

    # Plotando gráfico de rosca
    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct=make_autopct(values), 
                                      startangle=90, explode=explode, shadow=shadow)
    
    # Definindo título
    title = kwargs['title'] if 'title' in kwargs else f'Gráfico de Pizza para a Variável ${col}$'
    ax.set_title(title, size=16, color='dimgrey')

    # Parâmetros de customização do gráfico gerado
    autotexts_size = kwargs['autotexts_size'] if 'autotexts_size' in kwargs else 14
    autotexts_color = kwargs['autotexts_color'] if 'autotexts_color' in kwargs else 'white'
    texts_size = kwargs['texts_size'] if 'texts_size' in kwargs else 14
    texts_color = kwargs['texts_color'] if 'texts_stexts_colorize' in kwargs else 'black'

    # Customizando rótulos
    plt.setp(autotexts, size=autotexts_size, color=autotexts_color)
    plt.setp(texts, size=texts_size, color=texts_color)

    # Verificando salvamento da imagem
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col}_piechart.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

def plot_double_donut_chart(df, col1, col2, **kwargs):
    """
    Função responsável por plotar um gráfico de rosca customizado para uma determinada coluna da base
    
    Parâmetros
    ----------
    :param df: base de dados utilizada na plotagem [type: pd.DataFrame]
    :param col1: nome da primeira coluna a ser analisada (outer) [type: string]
    :param col1: nome da segunda coluna a ser analisada (inner) [type: string]
    :param **kwargs: parâmetros adicionais da função
        :arg label_names_col1: lista com rótulos da primeira coluna [type: list, default=value_counts().index]
        :arg label_names_col2: lista com rótulos da segunda coluna [type: list, default=value_counts().index]
        :arg figsize: dimensões da figura de plotagem [type: tuple, default=(8, 8)]
        :arg circle_radius: raio do círculo central do gráfico [type: float, default=0.55]
        :arg colors: lista de cores para aplicação na plotagem [type: list]
        :arg text: texto central do gráfico de rosca [type: string, default=f'Total: \n{sum(values)}']
        :arg title: título do gráfico [type: string, default=f'Gráfico Duplo de Rosca para ${col1}$ e ${col2}$']
        :arg autotexts_size: dimensão do rótulo do valor numérico do gráfico [type: int, default=14]
        :arg autotexts_color: cor do rótulo do valor numérico do gráfico [type: int, default='black]
        :arg texts_size: dimensão do rótulo do label [type: int, default=14]
        :arg texts_color: cor do rótulo do label [type: int, default='black']
        :arg save: flag indicativo de salvamento da imagem gerada [type: bool, default=None]
        :arg output_path: caminho de output da imagem a ser salva [type: string, default='output/']
        :arg img_name: nome do arquivo .png a ser gerado [type: string, default=f'{col}_donutchart.png']
     
    Retorno
    -------
    Essa função não retorna nenhum parâmetro além da plotagem customizada do gráfico duplo de rosca

    Aplicação
    ---------
    plot_donut_chart(df=df, col1='categorical_column', col2='categorical_column2)
    """
    
    # Validando presença da coluna na base
    if col1 not in df.columns:
        print(f'Coluna {col1} não presente na base')
        return
    if col2 not in df.columns:
        print(f'Coluna {col2} não presente na base')
        return

    # Retornando valores e labels para as duas camadas do gráfico
    first_layer_donut = df.groupby(col1).count().iloc[:, 0]
    first_layer_values = first_layer_donut.values
    second_layer_donut = df.groupby([col1, col2]).count().iloc[:, 0]
    second_layer_values = second_layer_donut.values
    col2_index = df.groupby(col2).count().iloc[:, 0].index
    
    # Criando DataFrame com dados da segunda camada
    second_layer_df = pd.DataFrame(second_layer_donut.index.values)
    second_layer_df['first_index'] = second_layer_df[0].apply(lambda x: x[0])
    second_layer_df['second_index'] = second_layer_df[0].apply(lambda x: x[1])
    second_layer_df['values'] = second_layer_donut.values

    # Retornando e mapeando labels para a legenda
    if 'label_names_col1' in kwargs:
        try:
            labels_col1 = first_layer_donut.index.map(kwargs['label_names_col1'])
        except Exception as e:
            print(f'Erro ao mapear o dicionário label_names_col1 na coluna {col1}. Exception: {e}')
    else:
        labels_col1 = first_layer_donut.index

    if 'label_names_col2' in kwargs:
        try:
            labels_col2 = second_layer_df['second_index'].map(kwargs['label_names_col2'])
        except Exception as e:
            print(f'Erro ao mapear o dicionário label_names_col2 na coluna {col2}. Exception: {e}')
    else:
        labels_col2 = second_layer_df['second_index']
    
    # Cores para a plotagem
    color_list = ['darkslateblue', 'crimson', 'lightseagreen', 'silver', 'lightskyblue', 'lightcoral']
    colors1 = kwargs['colors1'] if 'colors1' in kwargs else color_list[:len(label_names_col1)]
    colors2 = kwargs['colors2'] if 'colors2' in kwargs else color_list[-len(col2_index):]

    # Parâmetros de plotagem
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (8, 8)
    circle_radius = kwargs['circle_radius'] if 'circle_radius' in kwargs else 0.55
    circle_radius_color = kwargs['circle_radius_color'] if 'circle_radius_color' in kwargs else 'white'

    # Plotando gráfico de rosca
    center_circle = plt.Circle((0, 0), circle_radius, color=circle_radius_color)
    fig, ax = plt.subplots(figsize=figsize)
    wedges1, texts1, autotexts1 = ax.pie(first_layer_values, colors=colors1, startangle=90, 
                                         autopct=make_autopct(first_layer_values), pctdistance=1.20)
    wedges2, texts2, autotexts2 = ax.pie(second_layer_values, radius=0.75, colors=colors2, startangle=90, 
                                         autopct=make_autopct(second_layer_values), pctdistance=0.55)
    ax.add_artist(center_circle)

    # Configurando argumentos do texto central
    text = kwargs['text'] if 'text' in kwargs else ''
    text_kwargs = dict(size=20, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **text_kwargs)
    
    # Definindo título
    title = kwargs['title'] if 'title' in kwargs else f'Gráfico Duplo de Rosca para ${col1}$ e ${col2}$'
    ax.set_title(title, size=16, color='dimgrey')

    # Parâmetros de customização do gráfico gerado
    autotexts_size = kwargs['autotexts_size'] if 'autotexts_size' in kwargs else 14
    autotexts_color = kwargs['autotexts_color'] if 'autotexts_color' in kwargs else 'black'
    texts_size = kwargs['texts_size'] if 'texts_size' in kwargs else 14
    texts_color = kwargs['texts_color'] if 'texts_stexts_colorize' in kwargs else 'black'

    # Customizando rótulos
    plt.setp(autotexts1, size=autotexts_size, color=autotexts_color)
    plt.setp(texts1, size=texts_size, color=texts_color)
    plt.setp(autotexts2, size=autotexts_size, color=autotexts_color)
    plt.setp(texts2, size=texts_size, color=texts_color)
    
    # Customizando legendas
    custom_lines = []
    for c1 in colors1:
        custom_lines.append(Line2D([0], [0], color=c1, lw=4))
    for c2 in colors2:
        custom_lines.append(Line2D([0], [0], color=c2, lw=4))
    all_labels = list(labels_col1) + list(np.unique(labels_col2))
    ax.legend(custom_lines, labels=all_labels, fontsize=12, loc='upper left')

    # Verificando salvamento da imagem
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col1}_{col2}_donutchart.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

def plot_countplot(df, col, **kwargs):
    """
    Função responsável por plotar um gráfico de barras de volumetrias (countplot)
    
    Parâmetros
    ----------
    :param df: base de dados utilizada na plotagem [type: pd.DataFrame]
    :param col: referência de coluna a ser plotada [type: string]
    :param **kwargs: parâmetros adicionais da função   
        :arg top: filtro de top categorias a serem plotadas [type: int, default=-1]
        :arg orient: horizontal ou vertical [type: string, default='h']
        :arg figsize: dimensões da figura de plotagem [type: tuple, default=(8, 8)]
        :arg label_names: labels personalizados para os rótulos [type: dict, default=value_counts().index]
        :arg order: flag para ordenação dos dados [type: bool, default=True]
        :arg hue: parâmetro hue para quebra de plotagem do método countplot [type: string, default=None]
        :arg palette: paleta de cores utilizada na plotagem [type: string, default='rainbow']
        :arg title: título do gráfico [type: string, default=f'Volumetria para a variável {col}']
        :arg size_title: tamanho do título [type: int, default=16]
        :arg size_label: tamanho do rótulo [type: int, default=14]
        :arg save: flag indicativo de salvamento da imagem gerada [type: bool, default=None]
        :arg output_path: caminho de output da imagem a ser salva [type: string, default='output/']
        :arg img_name: nome do arquivo .png a ser gerado [type: string, default=f'{col}_countplot.png']
    
    Retorno
    -------
    Essa função não retorna nenhum parâmetro além de uma plotagem de volumetrias (barras)

    Aplicação
    ---------
    plot_countplot(df=df, col='column')
    """
    
    # Validando presença da coluna na base
    if col not in df.columns:
        print(f'Coluna {col} não presente na base')
        return
    
    # Retornando parâmetros de filtro de colunas
    top = kwargs['top'] if 'top' in kwargs else -1
    if top > 0:
        cat_count = df[col].value_counts()
        top_categories = cat_count[:top].index
        df = df[df[col].isin(top_categories)]
        
    # Parâmetros de plotagem
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (10, 7)
    hue = kwargs['hue'] if 'hue' in kwargs else None
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow'
    order = df[col].value_counts().index if 'order' in kwargs and bool(kwargs['order']) else None
    orient = kwargs['orient'] if 'orient' in kwargs and kwargs['orient'] in ['h', 'v'] else 'h'
        
    # Definindo orientação
    if orient == 'v':
        x = None
        y = col
    else:
        x = col
        y = None
    
    # Criando figura e aplicando countplot
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(data=df, ax=ax, x=x, y=y, hue=hue, order=order, palette=palette)

    # Retornando parâmetros de formatação da plotagem
    title = kwargs['title'] if 'title' in kwargs else f'Volumetria de Dados por {col}'
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 16
    size_labels = kwargs['size_labels'] if 'size_labels' in kwargs else 14
    label_names = kwargs['label_names'] if 'label_names' in kwargs else None
    
    # Formatando plotagem
    ax.set_title(title, size=size_title, pad=20)
    format_spines(ax, right_border=False)

    # Inserindo rótulo de percentual e modificando labels
    ncount = len(df)
    if x:
        # Rótulos
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            try:
                ax.annotate('{}\n{:.1f}%'.format(int(y), 100. * y / ncount), (x.mean(), y), 
                            ha='center', va='bottom', size=size_labels)
            except ValueError as ve: # Erro por divisão por zero em entradas inexistentes pela quebra
                continue
        
        # Labels
        if 'label_names' in kwargs:
            labels_old = ax.get_xticklabels()
            labels = [l.get_text() for l in labels_old]
            try:
                # Convertendo textos antes do mapeamento
                if type(list(kwargs['label_names'].keys())[0]) is int:
                    labels = [int(l) for l in labels]
                elif type(list(kwargs['label_names'].keys())[0]) is float:
                    labels = [float(l) for l in labels]
                
                # Mapeando rótulos customizados
                labels = pd.DataFrame(labels)[0].map(kwargs['label_names'])
                ax.set_xticklabels(labels)
            except Exception as e:
                print(f'Erro ao mapear labels na coluna {col}. Exception: {e}')
    else:
        # Rótulos
        for p in ax.patches:
            x = p.get_bbox().get_points()[1, 0]
            y = p.get_bbox().get_points()[:, 1]
            try:
                ax.annotate('{} ({:.1f}%)'.format(int(x), 100. * x / ncount), (x, y.mean()), 
                            va='center', size=size_labels)
            except ValueError as ve: # Erro por divisão por zero em entradas inexistentes pela quebra
                continue

        # Labels
        if 'label_names' in kwargs:
            labels_old = ax.get_yticklabels()
            labels = [l.get_text() for l in labels_old]
            try:
                # Convertendo textos antes do mapeamento
                if type(list(kwargs['label_names'].keys())[0]) is int:
                    labels = [int(l) for l in labels]
                elif type(list(kwargs['label_names'].keys())[0]) is float:
                    labels = [float(l) for l in labels]
                
                # Mapeando rótulos customizados
                labels = pd.DataFrame(labels)[0].map(kwargs['label_names'])
                ax.set_yticklabels(labels)
            except Exception as e:
                print(f'Erro ao mapear labels na coluna {col}. Exception: {e}')

    # Verificando salvamento da imagem
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col}_countplot.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

def plot_pct_countplot(df, col, hue, **kwargs):
    """
    Função responsável por plotar um gráfico de barras agrupadas com percentuais representativos
    
    Parâmetros
    ----------
    :param df: base de dados utilizada na plotagem [type: pd.DataFrame]
    :param col: referência de coluna a ser plotada [type: string]
    :param hue: parâmetro hue para quebra de plotagem [type: string]
    :param **kwargs: parâmetros adicionais da função   
        :arg top: filtro de top categorias a serem plotadas [type: int, default=-1]
        :arg orient: horizontal ou vertical [type: string, default='h']
        :arg figsize: dimensões da figura de plotagem [type: tuple, default=(8, 8)]
        :arg label_names: labels personalizados para os rótulos [type: dict, default=value_counts().index]
        :arg palette: paleta de cores utilizada na plotagem [type: string, default='rainbow']
        :arg title: título do gráfico [type: string, default=f'Volumetria para a variável {col}']
        :arg size_title: tamanho do título [type: int, default=16]
        :arg save: flag indicativo de salvamento da imagem gerada [type: bool, default=None]
        :arg output_path: caminho de output da imagem a ser salva [type: string, default='output/']
        :arg img_name: nome do arquivo .png a ser gerado [type: string, default=f'{col}_pct_countplot.png']
    
    Retorno
    -------
    Essa função não retorna nenhum parâmetro além de uma plotagem de representatividade por grupo

    Aplicação
    ---------
    plot_countplot(df=df, col='column')
    """
    
    # Validando presença da coluna na base
    if col not in df.columns:
        print(f'Coluna {col} não presente na base')
        return
    
    # Validando presença da coluna hue na base
    if hue not in df.columns:
        print(f'Coluna {hue} não presente na base')
        return
    
    # Retornando parâmetros de filtro de colunas
    top = kwargs['top'] if 'top' in kwargs else -1
    if top > 0:
        cat_count = df[col].value_counts()
        top_categories = cat_count[:top].index
        df = df[df[col].isin(top_categories)]
        
    # Retornando parâmetros de plotagem
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (10, 7)
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow'
    kind = 'bar' if 'orient' in kwargs and kwargs['orient'] == 'v' else 'barh'
    title = kwargs['title'] if 'title' in kwargs else f'Representatividade de {hue} para a coluna {col}'
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 16
    
    # Realizando quebra agrupada das colunas
    fig, ax = plt.subplots(figsize=figsize)
    col_to_hue = pd.crosstab(df[col], df[hue])
    col_to_hue.div(col_to_hue.sum(1).astype(float), axis=0).plot(kind=kind, stacked=True, ax=ax, 
                                                                 colormap=palette)
    
    # Customizando gráfico
    ax.set_title(title, size=size_title, pad=20)

    # Customizando rótulos
    if kind == 'barh':
        if 'label_names' in kwargs:
            labels_old = ax.get_xticklabels()
            labels = [l.get_text() for l in labels_old]
            try:
                # Convertendo textos antes do mapeamento
                if type(list(kwargs['label_names'].keys())[0]) is int:
                    labels = [int(l) for l in labels]
                elif type(list(kwargs['label_names'].keys())[0]) is float:
                    labels = [float(l) for l in labels]
                
                # Mapeando rótulos customizados
                labels = pd.DataFrame(labels)[0].map(kwargs['label_names'])
                ax.set_xticklabels(labels)
            except Exception as e:
                print(f'Erro ao mapear labels na coluna {col}. Exception: {e}')
    else:
        if 'label_names' in kwargs:
            labels_old = ax.get_yticklabels()
            labels = [l.get_text() for l in labels_old]
            try:
                # Convertendo textos antes do mapeamento
                if type(list(kwargs['label_names'].keys())[0]) is int:
                    labels = [int(l) for l in labels]
                elif type(list(kwargs['label_names'].keys())[0]) is float:
                    labels = [float(l) for l in labels]
                
                # Mapeando rótulos customizados
                labels = pd.DataFrame(labels)[0].map(kwargs['label_names'])
                ax.set_yticklabels(labels)
            except Exception as e:
                print(f'Erro ao mapear labels na coluna {col}. Exception: {e}')

    # Verificando salvamento da imagem
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col}_{hue}_pctcountplot.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

def plot_aggregation(df, group_col, value_col, aggreg, **kwargs):
    """
    Função responsável por plotagem de gráficos de agregação em barras
    
    Parâmetros
    ----------
    :param df: base de dados utilizada na plotagem [type: pd.DataFrame]
    :param group_col: coluna pivot de agrupamento [type: string]
    :param value_col: coluna com os valores a serem agregados [type: string]
    :param aggreg: informação da agregação utilizada na análise [type: string]
    :param **kwargs: parâmetros adicionais da função   
        :arg hue: parâmetro hue para quebra de plotagem do método countplot [type: string, default=None]
        :arg figsize: dimensões da figura de plotagem [type: tuple, default=(8, 8)]
        :arg top: filtro de top categorias a serem plotadas [type: int, default=-1]
        :arg orient: horizontal ou vertical [type: string, default='h']
        :arg label_names: labels personalizados para os rótulos [type: dict, default=value_counts().index]
        :arg palette: paleta de cores utilizada na plotagem [type: string, default='rainbow']
        :arg title: título do gráfico [type: string, default=f'Volumetria para a variável {col}']
        :arg size_title: tamanho do título [type: int, default=16]
        :arg size_label: t    hue = kwargs['hue'] if 'hue' in kwargs and kwargs['hue'] is not Noneamanho do rótulo [type: int, default=14]
        :arg save: flag indicativo de salvamento da imagem gerada [type: bool, default=None]
        :arg output_path: caminho de output da imagem a ser salva [type: string, default='output/']
        :arg img_name: nome do arquivo .png a ser gerado [type: string, default=f'{col}_countplot.png']
    
    Retorno
    -------
    Essa função não retorna nenhum parâmetro além de um gráfico de barras summarizado

    Aplicação
    ---------
    plot_aggregation(df=df, group_col='group', value_col='value', aggreg='agg')
    """
    
    # Verificando presença das colunas na base
    hue = kwargs['hue'] if 'hue' in kwargs else None
    df_columns = df.columns
    if group_col not in df_columns:
        print(f'Coluna {group_col} não presente na base')
        return
    if value_col not in df_columns:
        print(f'Coluna {value_col} não presente na base')
        return
    if hue is not None and hue not in df_columns:
        print(f'Coluna {hue} não presente na base')
        return

    # Aplicando agregação configurada
    try:
        if hue is not None:
            df_group = df.groupby(by=[group_col, hue], as_index=False).agg({value_col: aggreg})
        else:
            df_group = df.groupby(by=group_col, as_index=False).agg({value_col: aggreg})
        df_group.sort_values(by=value_col, ascending=False, inplace=True)
    except AttributeError as ae:
        print(f'Erro ao aplicar agregação com {aggreg}. Excepion lançada: {ae}')
        
    # Filtrando entradas
    if 'top' in kwargs and kwargs['top'] > 0:
        df_group = df_group[:kwargs['top']]
        
    # Retornando parâmetros de plotagem
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (10, 7)
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow'
    
    # Rótulos de medida para a plotagem
    if 'label_names' in kwargs:
        df_group[group_col] = df_group[group_col].map(kwargs['label_names'])
        
    # Orientação da plotagem
    orient = kwargs['orient'] if 'orient' in kwargs and kwargs['orient'] in ['h', 'v'] else 'v'
    if orient == 'v':
        x = group_col
        y = value_col
    else:
        x = value_col
        y = group_col
        
    # Retornando parâmetros de formatação da plotagem
    title = kwargs['title'] if 'title' in kwargs else f'Agrupamento de {group_col} por {aggreg} de {value_col}'
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 16
    size_labels = kwargs['size_labels'] if 'size_labels' in kwargs else 14
    
    # Construindo plotagem
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=x, y=y, data=df_group, hue=hue, palette=palette, ci=None, orient=orient)
    
    # Formatando plotagem
    ax.set_title(title, size=size_title, pad=20)
    format_spines(ax, right_border=False)

    # Inserindo rótulo de percentual
    if orient == 'h':
        AnnotateBars(n_dec=1, font_size=size_labels, color='black').horizontal(ax)
    else:
        AnnotateBars(n_dec=1, font_size=size_labels, color='black').vertical(ax)
            
    # Verificando salvamento da imagem
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{value_col}{hue}_{aggreg}plot_by{group_col}.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

def plot_distplot(df, col, kind='dist', **kwargs):
    """
    Função responsável pela plotagem de variáveis contínuas em formato de distribuição
    
    Parâmetros
    ----------
    :param df: base de dados utilizada na plotagem [type: pd.DataFrame]
    :param col: referência de coluna, de preferência numérica, a ser analisada [type: string]
    :param kind: tipo de plotagem de distribuição [type: string, default='dist']
        *opções: ['dist', 'kde', 'box', 'boxen', 'strip']
    :param **kwargs: parâmetros adicionais da função   
        :arg hue: parâmetro hue para quebra de plotagem do método countplot [type: string, default=None]
        :arg figsize: dimensões da figura de plotagem [type: tuple, default=(8, 8)]
        :arg label_names: labels personalizados para os rótulos [type: dict, default=value_counts().index]
        :arg palette: paleta de cores utilizada na plotagem [type: string, default='rainbow']
        :arg title: título do gráfico [type: string, default=f'Volumetria para a variável {col}']
        :arg size_title: tamanho do título [type: int, default=16]
        :arg save: flag indicativo de salvamento da imagem gerada [type: bool, default=None]
        :arg output_path: caminho de output da imagem a ser salva [type: string, default='output/']
        :arg img_name: nome do arquivo .png a ser gerado [type: string, default=f'{col}_countplot.png']
    
    Retorno
    -------
    Essa função não retorna nenhum parâmetro além de um gráfico de barras summarizado

    Aplicação
    ---------
    plot_distplot(df=df, col='column_name')
    """

    # Verificando presença das colunas na base
    hue = kwargs['hue'] if 'hue' in kwargs else None
    if col not in df.columns:
        print(f'Coluna {col} não presente na base')
        return
    if hue is not None and hue not in df.columns:
        print(f'Coluna {hue} não presente na base')
        return
    
    # Validando tipo de plotagem
    possible_kinds = ['dist', 'kde', 'box', 'boxen', 'strip']
    if kind not in possible_kinds:
        print(f'Parâmetro kind inválido. Opções possívels: {possible_kinds}')

    # Parâmetros de plotagem
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (10, 7)
    hist = kwargs['hist'] if 'hist' in kwargs else False
    kde = kwargs['kde'] if 'kde' in kwargs else True
    rug = kwargs['rug'] if 'rug' in kwargs else False
    shade = kwargs['shade'] if 'shade' in kwargs else True
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow'
    title = kwargs['title'] if 'title' in kwargs else f'{kind.title()}plot para a Variável {col}'
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 16

    sns.set(style='white', palette='muted', color_codes=True)

    # Construindo plotagem
    fig, ax = plt.subplots(figsize=figsize)
    
    # Distplot
    if kind == 'dist':
        if hue is not None:
            for cat in df[hue].value_counts().index:
                sns.distplot(df[df[hue]==cat][col], ax=ax, hist=hist, kde=kde, rug=rug, label=cat)
        else:
            sns.distplot(df[col], ax=ax, hist=hist, kde=kde, rug=rug)
    # Kdeplot        
    elif kind == 'kde':
        if hue is not None:
            for cat in df[hue].value_counts().index:
                sns.kdeplot(df[df[hue]==cat][col], ax=ax, shade=shade, label=cat)
        else:
            sns.kdeplot(df[col], ax=ax, shade=shade)
    # Boxplot
    elif kind == 'box':
        if hue is not None:
            sns.boxplot(x=hue, y=col, data=df, ax=ax, palette=palette)
        else:
            sns.boxplot(y=col, data=df, ax=ax, palette=palette)
    # Boxenplot
    elif kind == 'boxen':
        if hue is not None:
            sns.boxenplot(x=hue, y=col, data=df, ax=ax, palette=palette)
        else:
            sns.boxenplot(y=col, data=df, ax=ax, palette=palette)
    # Stripplot
    elif kind == 'strip':
        if hue is not None:
            sns.stripplot(x=hue, y=col, data=df, ax=ax, palette=palette)
        else:
            sns.stripplot(y=col, data=df, ax=ax, palette=palette)
            
    # Modificando labels
    if 'label_names' in kwargs and hue is not None and kind in ['box', 'boxen', 'strip']:
        labels_old = ax.get_xticklabels()
        labels = [l.get_text() for l in labels_old]
        try:
            # Convertendo textos antes do mapeamento
            if type(list(kwargs['label_names'].keys())[0]) is int:
                labels = [int(l) for l in labels]
            elif type(list(kwargs['label_names'].keys())[0]) is float:
                labels = [float(l) for l in labels]

            # Mapeando rótulos customizados
            labels = pd.DataFrame(labels)[0].map(kwargs['label_names'])
            ax.set_xticklabels(labels)
        except Exception as e:
            print(f'Erro ao mapear labels na coluna {col}. Exception: {e}')
            
    # Customizando gráfico
    format_spines(ax=ax, right_border=False)
    ax.set_title(title, size=size_title)
    if kind in ['dist', 'kde'] and hue is not None:
        ax.legend(title=hue)
        
    # Verificando salvamento da imagem
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col}{hue}_{kind}plot.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

def plot_corr_matrix(df, corr_col, corr='positive', **kwargs):
    """
    Função responsável por analisar correlação entre variáveis numéricas de uma base
    
    Parâmetros
    ----------
    :param df: base de dados utilizada na plotagem [type: pd.DataFrame]
    :param corr_col: referência de coluna alvo da correlação analisada [type: string]
    :param corr: tipo de correlação (positiva ou negativa) [type: string, default='positive']
    :param **kwargs: parâmetros adicionais da função   
        :arg figsize: dimensões da figura de plotagem [type: tuple, default=(8, 8)]
        :arg palette: paleta de cores utilizada na plotagem [type: string, default='YlGnBu' ou 'magma']
        :arg n_vars: quantidade de variáveis utilizadas na análise [type: int, default=10]
        :arg fmt: formato dos números/labels da matriz [type: string, default='.2f']
        :arg cbar: flag para plotagem da barra lateral de cores [type: bool, default=True]
        :arg annot: flag para anotação dos labels na matriz [type: bool, default=True]
        :arg square: flag para redimensionamento da matriz [type: bool, default=True]
        :arg title: título do gráfico [type: string, default=f'Análise de correlação de variáveis']
        :arg size_title: tamanho do título [type: int, default=16]
        :arg save: flag indicativo de salvamento da imagem gerada [type: bool, default=None]
        :arg output_path: caminho de output da imagem a ser salva [type: string, default='output/']
        :arg img_name: nome do arquivo .png a ser gerado [type: string, default=f'{col}_countplot.png']
    
    Retorno
    -------
    Essa função não retorna nenhum parâmetro além da matriz de correlação especificada

    Aplicação
    ---------
    plot_corr_matrix(df=df, corr_col='target')
    """
    
    # Criando matriz de correlação para a base de dados
    corr_mx = df.corr()
    
    # Retornando parâmetro de filtro de colunas
    num_features = [col for col, dtype in df.dtypes.items() if dtype != 'object']
    n_vars = kwargs['n_vars'] if 'n_vars' in kwargs else df[num_features].shape[1]

    # Verificando tipo de correlação e aplicando modificações
    if corr == 'positive':
        # Retornando top colunas com correlação positiva em relação a corr_col
        corr_cols = list(corr_mx.nlargest(n_vars+1, corr_col)[corr_col].index)
        pos_title = f'Top {n_vars} Features - Correlação Positiva \ncom a Variável ${corr_col}$'
        pos_cmap = 'YlGnBu'
        
        # Associando parâmetros gerais da função
        title = kwargs['title'] if 'title' in kwargs else pos_title
        cmap = kwargs['cmap'] if 'cmap' in kwargs else pos_cmap
        
    elif corr == 'negative':
        # Retornando top colunas com correlação negativa em relação a corr_col
        corr_cols = list(corr_mx.nsmallest(n_vars+1, corr_col)[corr_col].index)
        corr_cols = [corr_col] + corr_cols[:-1]
        neg_title = f'Top {n_vars} Features - Correlação Negativa \ncom a Variável ${corr_col}$'
        neg_cmap = 'magma'
        
        # Associando parâmetros gerais da função
        title = kwargs['title'] if 'title' in kwargs else neg_title
        cmap = kwargs['cmap'] if 'cmap' in kwargs else neg_cmap
        
    # Construindo array de correlação
    corr_data = np.corrcoef(df[corr_cols].values.T)

    # Retornando parâmetros gerais de plotagem
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (10, 10)
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 16
    cbar = kwargs['cbar'] if 'cbar' in kwargs else True
    annot = kwargs['annot'] if 'annot' in kwargs else True
    square = kwargs['square'] if 'square' in kwargs else True
    fmt = kwargs['fmt'] if 'fmt' in kwargs else '.2f'
    
    # Plotando matriz através de um heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_data, ax=ax, cbar=cbar, annot=annot, square=square, fmt=fmt, cmap=cmap,
                yticklabels=corr_cols, xticklabels=corr_cols)
    ax.set_title(title, size=size_title, color='black', pad=20)

def data_overview(df, **kwargs):
    """
    Análise geral em uma base de dados, retornando um DataFrame como resultado
    
    Parâmetros
    ----------
    :param df: base de dados utilizada na análise [type: pd.DataFrame]
    :param **kwargs: argumentos adicionais da função
        :arg corr: flag para análise de correlação [type: bool, default=False]
        :arg corr_method: método de correlação utilizada ('pearson', 'kendall', 'spearman')
        :arg target: variável target utilizada na correlação [type: string, default=None]
        :arg thresh_corr: limiar para filtro corr >= threshold [type: float, default=None]
        :arg thresh_null: limiar para filtro qtd_null >= threshold [type: float, default=0]
        :arg thresh_pct_null: limiar para filtro pct_null >= threshold [type: float, default=0]
        :arg sort: coluna de ordenação da base final [type: string, defaul='qtd_null']
        :arg ascending: flag de ordenação ascendente [type: bool, default=False]
        
    Retorno
    -------
    :return df_overview: DataFrame com análise das colunas das base
    
    Aplicação
    ---------
    df_overview = data_overview(df=df, corr=True, target='target')
    """
    
    # Retornando dados nulos
    df_null = pd.DataFrame(df.isnull().sum()).reset_index()
    df_null.columns = ['feature', 'qtd_null']
    df_null['pct_null'] = df_null['qtd_null'] / len(df)
    
    # Retornando tipo primitivo e quantidade de entradas dos atributos categóricos
    df_null['dtype'] = df_null['feature'].apply(lambda x: df[x].dtype)
    df_null['qtd_cat'] = [len(df[col].value_counts()) if df[col].dtype == 'object' else 0 for col in 
                          df_null['feature'].values]
    
    # Retornando parâmetros adicionais da função
    corr = kwargs['corr'] if 'corr' in kwargs else False
    corr_method = kwargs['corr_method'] if 'corr_method' in kwargs else 'pearson'
    target = kwargs['target'] if 'target' in kwargs else None
    thresh_corr = kwargs['thresh_corr'] if 'thresh_corr' in kwargs else None
    thresh_null = kwargs['thresh_null'] if 'thresh_null' in kwargs else 0
    thresh_pct_null = kwargs['thresh_pct_null'] if 'thresh_pct_null' in kwargs else 0   
    sort = kwargs['sort'] if 'sort' in kwargs else 'qtd_null'
    ascending = kwargs['sort_ascending'] if 'sort_ascending' in kwargs else False
    
    # Verificando análise de correlação
    if corr and target is None:
        print(f'Ao definir "correlation=True" é preciso também definir o argumento "target"')
    
    if corr and target is not None:
        # Extraindo correlação especificada
        target_corr = pd.DataFrame(df.corr(method=corr_method)[target])
        target_corr = target_corr.reset_index()
        target_corr.columns = ['feature', f'target_{corr_method}_corr']
        
        # Aplicando join
        df_overview = df_null.merge(target_corr, how='left', on='feature')
        if thresh_corr is not None:
            df_overview = df_overview[df_overview[f'target_{corr_method}_corr'] > thresh_corr]
            
    else:
        # Análise de correlação não será aplicada
        df_overview = df_null
        
    # Filtrando dados nulos
    df_overview = df_overview.query('pct_null >= @thresh_null')
    df_overview = df_overview.query('qtd_null >= @thresh_pct_null')
    
    # Ordenando base
    df_overview.sort_values(by=sort, ascending=ascending, inplace=True)
    df_overview.reset_index(drop=True, inplace=True)
    
    return df_overview

    