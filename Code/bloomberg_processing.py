"""
bloomberg_processing.py

Processamento de dados reais da Bloomberg Terminal para criação de datasets
completos de WTI e Brent com estrutura de termo real (não sintética).

OBJETIVOS:
- Ler dados brutos da Bloomberg (BRENT 2010 ate 2025.xlsx e WTI 2010 ate 2025.xlsx)
- Fazer análise exploratória da estrutura dos dados
- Criar datasets WTI_bloomberg e Brent_bloomberg formatados para o pipeline
- Manter dados reais sem interpolações artificiais
- Gerar estrutura compatível com nosso sistema

FLUXO PRINCIPAL:
1. Análise exploratória dos dados brutos
2. Identificação de contratos e maturidades
3. Limpeza e formatação dos dados
4. Criação de datasets finais (WTI_bloomberg e Brent_bloomberg)

AUTOR: Sistema ItauQuant
DATA: 2025-11-02
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BloombergProcessor:
    """
    Processador de dados da Bloomberg Terminal para commodities.
    """
    
    def __init__(self, raw_data_path: str = "data/raw/Blumberg"):
        """
        Inicializa o processador.
        
        Parameters
        ----------
        raw_data_path : str
            Caminho para a pasta com dados brutos da Bloomberg
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path("data/raw")
        
        logger.info(f"🔍 Bloomberg Processor inicializado")
        logger.info(f"  📁 Dados brutos: {self.raw_data_path}")
        logger.info(f"  📤 Saída: {self.output_path}")
    
    def explore_raw_data(self) -> Dict[str, any]:
        """
        Análise exploratória dos dados brutos da Bloomberg.
        
        Returns
        -------
        Dict[str, any]
            Relatório da análise exploratória
        """
        print("=" * 80)
        print("🔍 ANÁLISE EXPLORATÓRIA DOS DADOS DA BLOOMBERG")
        print("=" * 80)
        
        exploration_report = {
            'files_found': [],
            'wti_analysis': {},
            'brent_analysis': {},
            'data_structure': {},
            'recommendations': []
        }
        
        # 1. Listar arquivos encontrados
        print("\n📁 1. ARQUIVOS DISPONÍVEIS:")
        
        wti_file = self.raw_data_path / "WTI 2010 ate 2025.xlsx"
        brent_file = self.raw_data_path / "BRENT 2010 ate 2025.xlsx"
        
        files_info = []
        for file_path in [wti_file, brent_file]:
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                files_info.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size_kb': size_kb,
                    'commodity': 'WTI' if 'WTI' in file_path.name else 'BRENT'
                })
                print(f"   ✅ {file_path.name} ({size_kb:.1f} KB)")
            else:
                print(f"   ❌ {file_path.name} - NÃO ENCONTRADO")
        
        exploration_report['files_found'] = files_info
        
        # 2. Análise detalhada do WTI
        if wti_file.exists():
            print(f"\n📊 2. ANÁLISE DETALHADA: WTI")
            wti_analysis = self._analyze_excel_file(wti_file, "WTI")
            exploration_report['wti_analysis'] = wti_analysis
        
        # 3. Análise detalhada do BRENT  
        if brent_file.exists():
            print(f"\n📊 3. ANÁLISE DETALHADA: BRENT")
            brent_analysis = self._analyze_excel_file(brent_file, "BRENT")
            exploration_report['brent_analysis'] = brent_analysis
        
        # 4. Resumo e recomendações
        print(f"\n💡 4. RESUMO E RECOMENDAÇÕES:")
        recommendations = self._generate_recommendations(exploration_report)
        exploration_report['recommendations'] = recommendations
        
        for rec in recommendations:
            print(f"   • {rec}")
        
        return exploration_report
    
    def _analyze_excel_file(self, file_path: Path, commodity: str) -> Dict[str, any]:
        """
        Análise detalhada de um arquivo Excel da Bloomberg.
        
        Parameters
        ----------
        file_path : Path
            Caminho para o arquivo Excel
        commodity : str
            Nome da commodity (WTI ou BRENT)
        
        Returns
        -------
        Dict[str, any]
            Análise detalhada do arquivo
        """
        analysis = {
            'sheets': [],
            'data_sample': {},
            'date_range': {},
            'columns': [],
            'data_types': {},
            'missing_data': {},
            'recommendations': []
        }
        
        try:
            # Ler informação das sheets
            xl_file = pd.ExcelFile(file_path)
            analysis['sheets'] = xl_file.sheet_names
            print(f"      📋 Sheets encontradas: {len(analysis['sheets'])}")
            for sheet in analysis['sheets']:
                print(f"         • {sheet}")
            
            # Analisar a primeira sheet (principal)
            main_sheet = analysis['sheets'][0] if analysis['sheets'] else None
            if main_sheet:
                print(f"      🔎 Analisando sheet principal: '{main_sheet}'")
                
                # Ler dados
                df = pd.read_excel(file_path, sheet_name=main_sheet)
                print(f"         📐 Shape: {df.shape} (linhas x colunas)")
                
                # Analisar colunas
                analysis['columns'] = df.columns.tolist()
                print(f"         📊 Colunas ({len(analysis['columns'])}):")
                for i, col in enumerate(analysis['columns'][:10]):  # Primeiras 10 colunas
                    print(f"            [{i}] {col}")
                if len(analysis['columns']) > 10:
                    print(f"            ... e mais {len(analysis['columns']) - 10} colunas")
                
                # Analisar tipos de dados
                analysis['data_types'] = df.dtypes.to_dict()
                print(f"         🔤 Tipos de dados:")
                for col, dtype in list(analysis['data_types'].items())[:5]:
                    print(f"            {col}: {dtype}")
                
                # Sample dos dados
                print(f"         📋 Primeiras 5 linhas:")
                sample_df = df.head()
                analysis['data_sample'] = sample_df.to_dict()
                print(sample_df.to_string())
                
                # Verificar se primeira coluna parece ser data
                first_col = df.columns[0]
                if 'date' in first_col.lower() or df[first_col].dtype in ['datetime64[ns]', 'object']:
                    print(f"         📅 Primeira coluna parece ser DATA: {first_col}")
                    
                    # Tentar converter para datetime
                    try:
                        dates = pd.to_datetime(df[first_col])
                        date_min = dates.min()
                        date_max = dates.max()
                        analysis['date_range'] = {
                            'start': str(date_min.date()) if pd.notna(date_min) else 'N/A',
                            'end': str(date_max.date()) if pd.notna(date_max) else 'N/A',
                            'total_days': len(dates.dropna())
                        }
                        print(f"            📆 Range: {analysis['date_range']['start']} → {analysis['date_range']['end']}")
                        print(f"            📊 Total de dias: {analysis['date_range']['total_days']}")
                    except:
                        print(f"            ❌ Não conseguiu parsear datas")
                
                # Missing data
                missing_counts = df.isnull().sum()
                analysis['missing_data'] = missing_counts.to_dict()
                print(f"         🕳️  Dados faltantes:")
                for col, missing in missing_counts.head().items():
                    pct = (missing / len(df)) * 100
                    print(f"            {col}: {missing} ({pct:.1f}%)")
                
                # Verificar se colunas parecem ser contratos futuros
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                print(f"         💰 Colunas numéricas ({len(numeric_cols)}):")
                for col in numeric_cols[:8]:  # Primeiras 8 colunas numéricas
                    sample_values = df[col].dropna()
                    if len(sample_values) > 0:
                        print(f"            {col}: min={sample_values.min():.2f}, max={sample_values.max():.2f}, mean={sample_values.mean():.2f}")
                
        except Exception as e:
            print(f"      ❌ Erro ao analisar {commodity}: {str(e)}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _generate_recommendations(self, exploration_report: Dict[str, any]) -> List[str]:
        """
        Gera recomendações baseadas na análise exploratória.
        
        Parameters
        ----------
        exploration_report : Dict[str, any]
            Relatório da análise exploratória
        
        Returns
        -------
        List[str]
            Lista de recomendações
        """
        recommendations = []
        
        # Verificar se ambos os arquivos foram encontrados
        files = exploration_report.get('files_found', [])
        wti_found = any('WTI' in f['name'] for f in files)
        brent_found = any('BRENT' in f['name'] for f in files)
        
        if wti_found and brent_found:
            recommendations.append("✅ Ambos arquivos (WTI e BRENT) encontrados")
        elif wti_found:
            recommendations.append("⚠️ Apenas WTI encontrado - BRENT faltando")
        elif brent_found:
            recommendations.append("⚠️ Apenas BRENT encontrado - WTI faltando")
        else:
            recommendations.append("❌ Nenhum arquivo encontrado")
        
        # Analisar estrutura de dados
        for commodity in ['wti', 'brent']:
            analysis_key = f'{commodity}_analysis'
            if analysis_key in exploration_report:
                analysis = exploration_report[analysis_key]
                
                if 'error' in analysis:
                    recommendations.append(f"❌ {commodity.upper()}: Erro na leitura - verificar formato")
                else:
                    if analysis.get('date_range', {}).get('total_days', 0) > 1000:
                        recommendations.append(f"✅ {commodity.upper()}: Dados históricos extensos ({analysis['date_range']['total_days']} dias)")
                    
                    numeric_cols = len([col for col in analysis.get('columns', []) if col not in ['Date', 'date', 'Dates']])
                    if numeric_cols >= 6:
                        recommendations.append(f"✅ {commodity.upper()}: Estrutura de termo rica ({numeric_cols} contratos)")
                    elif numeric_cols >= 3:
                        recommendations.append(f"⚠️ {commodity.upper()}: Estrutura limitada ({numeric_cols} contratos)")
                    else:
                        recommendations.append(f"❌ {commodity.upper()}: Poucos contratos ({numeric_cols})")
        
        recommendations.append("🔄 Próximo passo: Executar create_datasets() para processar os dados")
        
        return recommendations
    
    def create_datasets(self) -> Dict[str, str]:
        """
        Cria datasets WTI e Brent formatados a partir dos dados da Bloomberg.
        
        Returns
        -------
        Dict[str, str]
            Caminhos dos datasets criados
        """
        print("=" * 80)
        print("🏗️ CRIANDO DATASETS FORMATADOS DA BLOOMBERG")
        print("=" * 80)
        
        created_datasets = {}
        
        # Processar WTI
        wti_path = self._process_wti_data()
        if wti_path:
            created_datasets['WTI_bloomberg'] = wti_path
        
        # Processar BRENT
        brent_path = self._process_brent_data()
        if brent_path:
            created_datasets['BRENT_bloomberg'] = brent_path
        
        print(f"\n🎉 DATASETS CRIADOS COM SUCESSO!")
        for name, path in created_datasets.items():
            print(f"   ✅ {name}: {path}")
        
        return created_datasets
    
    def _process_wti_data(self) -> Optional[str]:
        """
        Processa dados do WTI e cria dataset formatado.
        
        Returns
        -------
        Optional[str]
            Caminho do dataset criado ou None se erro
        """
        print(f"\n🛢️ PROCESSANDO WTI...")
        
        try:
            wti_file = self.raw_data_path / "WTI 2010 ate 2025.xlsx"
            
            # Ler dados brutos
            df_raw = pd.read_excel(wti_file, sheet_name='Sheet1')
            print(f"   📊 Dados brutos: {df_raw.shape}")
            
            # Analisar estrutura mais detalhadamente
            print(f"   🔍 Análise da estrutura:")
            for i, col in enumerate(df_raw.columns):
                sample_val = df_raw[col].dropna().iloc[0] if len(df_raw[col].dropna()) > 0 else "N/A"
                print(f"      [{i}] {col}: {df_raw[col].dtype} | Sample: {sample_val}")
            
            # Identificar colunas de datas e preços
            date_cols = []
            price_cols = []
            
            for col in df_raw.columns:
                if df_raw[col].dtype == 'datetime64[ns]':
                    date_cols.append(col)
                elif df_raw[col].dtype in ['float64', 'int64'] and df_raw[col].notna().sum() > 100:
                    # Verificar se parece preço de petróleo (entre 10 e 200)
                    sample_values = df_raw[col].dropna()
                    if len(sample_values) > 0:
                        if 10 <= sample_values.mean() <= 200:
                            price_cols.append(col)
            
            print(f"   📅 Colunas de data identificadas: {date_cols}")
            print(f"   💰 Colunas de preço identificadas: {price_cols}")
            
            # Assumir que dados estão organizados em pares (data, preço)
            # Baseado na análise: CL1 (data), Unnamed: 4 (preço CL1), C01 (data), Unnamed: 7 (preço C01)
            
            contracts_data = []
            
            # Processar contratos identificados
            if 'CL1' in df_raw.columns and 'Unnamed: 4' in df_raw.columns:
                cl1_data = self._extract_contract_data(df_raw, 'CL1', 'Unnamed: 4', 'CL1')
                if cl1_data is not None:
                    contracts_data.append(('tenor_1', cl1_data))
            
            if 'C01 ' in df_raw.columns and 'Unnamed: 7' in df_raw.columns:
                c01_data = self._extract_contract_data(df_raw, 'C01 ', 'Unnamed: 7', 'C01')
                if c01_data is not None:
                    contracts_data.append(('tenor_2', c01_data))
            
            print(f"   ✅ Contratos extraídos: {len(contracts_data)}")
            
            if len(contracts_data) < 2:
                print(f"   ❌ Poucos contratos extraídos para criar dataset válido")
                return None
            
            # Criar estrutura final
            dataset_path = self._build_dataset('WTI_bloomberg', contracts_data, 'WTI')
            return dataset_path
            
        except Exception as e:
            print(f"   ❌ Erro ao processar WTI: {str(e)}")
            return None
    
    def _process_brent_data(self) -> Optional[str]:
        """
        Processa dados do BRENT e cria dataset formatado.
        
        Returns
        -------
        Optional[str]
            Caminho do dataset criado ou None se erro
        """
        print(f"\n🛢️ PROCESSANDO BRENT...")
        
        try:
            brent_file = self.raw_data_path / "BRENT 2010 ate 2025.xlsx"
            
            # Ler dados brutos
            df_raw = pd.read_excel(brent_file, sheet_name='Sheet1')
            print(f"   📊 Dados brutos: {df_raw.shape}")
            
            # Analisar estrutura
            print(f"   🔍 Análise da estrutura:")
            for i, col in enumerate(df_raw.columns):
                sample_val = df_raw[col].dropna().iloc[0] if len(df_raw[col].dropna()) > 0 else "N/A"
                print(f"      [{i}] {col}: {df_raw[col].dtype} | Sample: {sample_val}")
            
            # Para BRENT, parece mais simples: Unnamed: 3 (data), BRENT (preço)
            contracts_data = []
            
            if 'Unnamed: 3' in df_raw.columns and 'BRENT' in df_raw.columns:
                brent_data = self._extract_contract_data(df_raw, 'Unnamed: 3', 'BRENT', 'BRENT')
                if brent_data is not None:
                    contracts_data.append(('tenor_1', brent_data))
            
            print(f"   ✅ Contratos extraídos: {len(contracts_data)}")
            
            if len(contracts_data) < 1:
                print(f"   ❌ Nenhum contrato válido extraído")
                return None
            
            # Criar estrutura final
            dataset_path = self._build_dataset('BRENT_bloomberg', contracts_data, 'BRENT')
            return dataset_path
            
        except Exception as e:
            print(f"   ❌ Erro ao processar BRENT: {str(e)}")
            return None
    
    def _extract_contract_data(self, df: pd.DataFrame, date_col: str, price_col: str, contract_name: str) -> Optional[pd.Series]:
        """
        Extrai dados de um contrato específico.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame com dados brutos
        date_col : str
            Nome da coluna de datas
        price_col : str
            Nome da coluna de preços
        contract_name : str
            Nome do contrato para logs
        
        Returns
        -------
        Optional[pd.Series]
            Série temporal com preços indexada por data
        """
        try:
            # Extrair dados válidos
            dates = pd.to_datetime(df[date_col]).dropna()
            prices = df[price_col].dropna()
            
            # Alinhar datas e preços (mesmo índice)
            valid_mask = df[date_col].notna() & df[price_col].notna()
            if valid_mask.sum() == 0:
                print(f"      ❌ {contract_name}: Nenhum dado válido")
                return None
            
            clean_dates = pd.to_datetime(df.loc[valid_mask, date_col])
            clean_prices = df.loc[valid_mask, price_col]
            
            # Criar série temporal
            series = pd.Series(clean_prices.values, index=clean_dates, name=contract_name)
            series = series.sort_index().dropna()
            
            print(f"      ✅ {contract_name}: {len(series)} pontos, {series.index[0].date()} → {series.index[-1].date()}")
            print(f"         Range: ${series.min():.2f} - ${series.max():.2f}, Média: ${series.mean():.2f}")
            
            return series
            
        except Exception as e:
            print(f"      ❌ {contract_name}: Erro - {str(e)}")
            return None
    
    def _build_dataset(self, dataset_name: str, contracts_data: List[Tuple[str, pd.Series]], commodity: str) -> str:
        """
        Constrói dataset formatado no padrão do sistema.
        
        Parameters
        ----------
        dataset_name : str
            Nome do dataset
        contracts_data : List[Tuple[str, pd.Series]]
            Lista de (nome_tenor, serie_temporal)
        commodity : str
            Nome da commodity
        
        Returns
        -------
        str
            Caminho do dataset criado
        """
        print(f"   🏗️ Construindo dataset {dataset_name}...")
        
        # Encontrar índice comum (interseção de todas as datas)
        all_indexes = [series.index for _, series in contracts_data]
        common_index = all_indexes[0]
        for idx in all_indexes[1:]:
            common_index = common_index.intersection(idx)
        
        print(f"      📅 Índice comum: {len(common_index)} dias")
        print(f"         Período: {common_index[0].date()} → {common_index[-1].date()}")
        
        # Construir F_mkt (preços futuros)
        F_mkt_dict = {}
        for tenor_name, series in contracts_data:
            F_mkt_dict[tenor_name] = series.reindex(common_index)
        
        F_mkt = pd.DataFrame(F_mkt_dict, index=common_index)
        
        # Construir ttm (time-to-maturity) - assumir tenores mensais
        ttm_dict = {}
        for i, (tenor_name, _) in enumerate(contracts_data):
            months = i + 1
            ttm_years = months / 12.0
            ttm_dict[tenor_name] = [ttm_years] * len(common_index)
        
        ttm = pd.DataFrame(ttm_dict, index=common_index)
        
        # Spot (usar primeiro contrato como proxy)
        S = pd.DataFrame({'S': F_mkt.iloc[:, 0]}, index=common_index)
        
        # Custos (padrão)
        costs = pd.Series([10.0] * len(F_mkt.columns), index=F_mkt.columns, name='cost_per_contract')
        
        # Salvar dataset
        dataset_path = self.output_path / dataset_name
        dataset_path.mkdir(exist_ok=True)
        
        # Salvar arquivos
        F_mkt.to_csv(dataset_path / 'F_mkt.csv')
        ttm.to_csv(dataset_path / 'ttm.csv')
        S.to_csv(dataset_path / 'S.csv')
        costs.to_csv(dataset_path / 'costs.csv')
        
        # Metadados
        info = {
            'source': 'Bloomberg Terminal',
            'commodity': commodity,
            'dataset_name': dataset_name,
            'created_at': datetime.now().isoformat(),
            'start_date': str(common_index[0].date()),
            'end_date': str(common_index[-1].date()),
            'num_days': len(common_index),
            'num_tenors': len(contracts_data),
            'tenors': [name for name, _ in contracts_data],
            'price_range': {
                'min': float(F_mkt.min().min()),
                'max': float(F_mkt.max().max()),
                'mean': float(F_mkt.mean().mean())
            }
        }
        
        with open(dataset_path / 'info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"      ✅ Dataset salvo: {dataset_path}")
        print(f"         📊 Shape: {F_mkt.shape}")
        print(f"         💰 Range de preços: ${info['price_range']['min']:.2f} - ${info['price_range']['max']:.2f}")
        
        return str(dataset_path)


def main():
    """
    Função principal para análise exploratória.
    """
    print("🚀 INICIANDO PROCESSAMENTO DOS DADOS DA BLOOMBERG")
    print("=" * 80)
    
    # Inicializar processador
    processor = BloombergProcessor()
    
    # Executar análise exploratória
    exploration_report = processor.explore_raw_data()
    
    # Salvar relatório
    report_path = Path("data/raw/bloomberg_exploration_report.json")
    with open(report_path, 'w') as f:
        json.dump(exploration_report, f, indent=2, default=str)
    
    print(f"\n💾 Relatório salvo em: {report_path}")
    print("\n🎯 PRÓXIMOS PASSOS:")
    print("   1. Revisar a análise acima")
    print("   2. Se tudo OK, executar: processor.create_datasets()")
    print("   3. Datasets WTI_bloomberg e Brent_bloomberg serão criados")


def create_bloomberg_datasets():
    """
    Função para criar os datasets formatados da Bloomberg.
    """
    print("🏗️ CRIANDO DATASETS DA BLOOMBERG")
    print("=" * 80)
    
    # Inicializar processador
    processor = BloombergProcessor()
    
    # Criar datasets
    created_datasets = processor.create_datasets()
    
    if created_datasets:
        print(f"\n🎉 SUCESSO! Datasets criados:")
        for name, path in created_datasets.items():
            print(f"   ✅ {name}")
            print(f"      📁 {path}")
        
        print(f"\n🚀 PRONTO PARA USAR NO SISTEMA!")
        print("   Execute o backtest com:")
        if 'WTI_bloomberg' in created_datasets:
            print("   python Code/backtest.py --dataset WTI_bloomberg")
        if 'BRENT_bloomberg' in created_datasets:
            print("   python Code/backtest.py --dataset BRENT_bloomberg")
    else:
        print(f"\n❌ Nenhum dataset foi criado. Verifique os logs acima.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        create_bloomberg_datasets()
    else:
        main()
