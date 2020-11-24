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
from pycomp.viz.formatador import make_autopct
import matplotlib.pyplot as plt
import os


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

def plot_donut_chart(df, col, figsize=(8, 8), circle_radius=0.8, **kwargs):
    """
    Função responsável por plotar um gráfico de rosca customizado para uma determinada coluna da base
    
    Parâmetros
    ----------
    :param df: base de dados utilizada na plotagem [type: pd.DataFrame]
    :param col: nome da coluna a ser analisada [type: string]
    :param figsize: dimensões da figura de plotagem [type: tuple, default=(8, 8)]
    :param circle_radius: raio do círculo central do gráfico [type: float, default=0.8]
    :param **kwargs: parâmetros adicionais da função
        :arg label_names: lista com labels personalizados para os rótulos [type: list, default=value_counts().index]
        :arg flag_ruido: índice de filtro para eliminar as n últimas entradas [type: float, default=None]
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
    plot_donut_chart(df=df, col='categorical_column', label_names=['Classe 1', 'Classe 2'])
    """
    
    # Retorno dos valores e definição da figura
    try:
        values = df[col].value_counts().values
    except KeyError as e:
        cat_cols = [col for col, dtype in df.dtypes.items() if dtype == 'object']
        print(f'Coluna "{col}" não presente na base. Colunas categóricas disponíveis: {cat_cols}')
        return
    
    # Rótulos de medida para a plotagem
    label_names = kwargs['label_names'] if 'label_names' in kwargs else df[col].value_counts().index
    
    # Verificando parâmetro de supressão de alguma categoria da análise
    if 'flag_ruido' in kwargs and kwargs['flag_ruido'] > 0:
        flag_ruido = kwargs['flag_ruido']
        values = values[:-flag_ruido]
        label_names = label_names[:-flag_ruido]
    
    # Cores para a plotagem
    color_list = ['darkslateblue', 'crimson', 'lightseagreen', 'lightskyblue', 'lightcoral', 'silver']
    colors = kwargs['colors'] if 'colors' in kwargs else color_list[:len(label_names)]

    # Plotando gráfico de rosca
    center_circle = plt.Circle((0, 0), circle_radius, color='white')
    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(values, labels=label_names, colors=colors, autopct=make_autopct(values))
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

        # Retornando diretório e nome da imagem
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col}_donutchart.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

def plot_pie_chart(df, col, figsize=(8, 8), **kwargs):
    """
    Função responsável por plotar um gráfico de pizza customizado para uma determinada coluna da base
    
    Parâmetros
    ----------
    :param df: base de dados utilizada na plotagem [type: pd.DataFrame]
    :param col: nome da coluna a ser analisada [type: string]
    :param figsize: dimensões da figura de plotagem [type: tuple, default=(8, 8)]
    :param **kwargs: parâmetros adicionais da função
        :arg label_names: lista com labels personalizados para os rótulos [type: list, default=value_counts().index]
        :arg flag_ruido: índice de filtro para eliminar as n últimas entradas [type: float, default=None]
        :arg colors: lista de cores para aplicação na plotagem [type: list]
        :arg title: título do gráfico [type: string, default=f'Gráfico de Rosca para a Variável ${col}$']
        :arg autotexts_size: dimensão do rótulo do valor numérico do gráfico [type: int, default=14]
        :arg autotexts_color: cor do rótulo do valor numérico do gráfico [type: int, default='white']
        :arg texts_size: dimensão do rótulo do label [type: int, default=14]
        :arg texts_color: cor do rótulo do label [type: int, default='black']
        :arg save: flag indicativo de salvamento da imagem gerada [type: bool, default=None]
        :arg output_path: caminho de output da imagem a ser salva [type: string, default='output/']
        :arg img_name: nome do arquivo .png a ser gerado [type: string, default=f'{col}_donutchart.png']
    
    
    Retorno
    -------
    Essa função não retorna nenhum parâmetro além da plotagem customizada do gráfico de pizaa

    Aplicação
    ---------
    plot_pie_chart(df=df, col='categorical_column', label_names=['Classe 1', 'Classe 2'])
    """
    
    # Retorno dos valores e definição da figura
    try:
        values = df[col].value_counts().values
    except KeyError as e:
        cat_cols = [col for col, dtype in df.dtypes.items() if dtype == 'object']
        print(f'Coluna "{col}" não presente na base. Colunas categóricas disponíveis: {cat_cols}')
        return
    
    # Rótulos de medida para a plotagem
    label_names = kwargs['label_names'] if 'label_names' in kwargs else df[col].value_counts().index
    
    # Verificando parâmetro de supressão de alguma categoria da análise
    if 'flag_ruido' in kwargs and kwargs['flag_ruido'] > 0:
        flag_ruido = kwargs['flag_ruido']
        values = values[:-flag_ruido]
        label_names = label_names[:-flag_ruido]
    
    # Cores para a plotagem
    color_list = ['darkslateblue', 'crimson', 'lightseagreen', 'lightskyblue', 'lightcoral', 'silver']
    colors = kwargs['colors'] if 'colors' in kwargs else color_list[:len(label_names)]

    # Parâmetros de plotagem do gráfico de pizza
    explode = kwargs['explode'] if 'explode' in kwargs else (0,) * len(label_names)
    shadow = kwargs['shadow'] if 'shadow' in kwargs else False

    # Plotando gráfico de pizza
    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(values, labels=label_names, colors=colors, autopct=make_autopct(values), 
                                      startangle=90, explode=explode, shadow=shadow)
    
    # Definindo título
    title = kwargs['title'] if 'title' in kwargs else f'Gráfico de Rosca para a Variável ${col}$'
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

        # Retornando diretório e nome da imagem
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col}_piechart.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)