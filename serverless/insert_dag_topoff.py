# coding: utf-8
########################################################################################################################
###########################################                       ######################################################
###########################################  READ ONLY LIBRARIES  ######################################################
###########################################                       ######################################################
########################################################################################################################
import json, logging, os, sys, re
sys.path.append(os.path.dirname(__file__))
log = logging.getLogger(__name__)
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators import MATInstanceInitOperator, MATInstanceExitOperator, MATPythonOperator, DummyOperator, MATBranchOperator, MATAnsibleBranchOperator, MATAnsiblePlaybookOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.exceptions import AirflowException, AirflowSkipException
from mat import *
from mat_error_management import MATWorkflowErrorManagement
from mat_success_management import MATWorkflowSuccessManagement
from mat_runtime import * 

#Codigo DM
#from ftplib import FTP
#import os
#import pysftp
########################################################################################################################
#######################################                                  ###############################################
#######################################  CUSTOM LIBRARIES AND VARIABLES  ###############################################
#######################################                                  ###############################################
########################################################################################################################

try:
    import usecase

except:
    pass

import paramiko
import pandas as pd
import gzip
import zipfile as zip
import openpyxl
from pathlib import Path
from airflow.utils.email import send_email
from sendmail import ATTSendEmail
import shutil
import glob
import gc
########################################################################################################################
##############################################                  ########################################################
##############################################  DAG DEFINITION  ########################################################
##############################################                  ########################################################
########################################################################################################################


default_args = {
    'owner': 'Iquall',
    'depends_on_past': False,
    'provide_context': True,
    'retries': 3,  
    'retry_delay': timedelta(minutes=5),

}

dag = DAG(dag_id='sdt-22i-cei', description='', start_date=datetime(2026,7,22), schedule_interval='*/5 * * * *', catchup=False, on_failure_callback=MATWorkflowErrorManagement, on_success_callback=MATWorkflowSuccessManagement, default_args=default_args)


########################################################################################################################
##############################################                    ######################################################
##############################################  CUSTOM FUNCTIONS  ######################################################
##############################################                    ######################################################
########################################################################################################################

## HELPERS ##
def get_sftp_credentials():
    inv = Inventory()
    cipher = crypto.aes256.MATCipher()

    host_data = inv.get(
        module="mathost",
        query="data.hostname=EPT_DM",
    )

    if not host_data:
        raise ValueError("No se encontró el host EPT_DM en el inventario.")

    sftp_data = host_data[0]["data"]

    credentials = sftp_data["networkRole"]["data"]["matcredentials"]["data"]

    username = credentials["username"]
    password = cipher.decrypt(credentials["password"])
    host = sftp_data["managementIp"]

    return host, username, password

def get_work_base_dir(kwargs):
    """
    Directorio dinamico por corrida.
    Todo lo temporal del DAG vive dentro de esta subcarpeta y se elimina al final.
    """
    base_dir = Path(MATDirectories(kwargs).run) / "dashboard_topoff_work"
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir)


def clean_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value
    
def is_dashboard_topoff_available(mat, job):
    query = """
    query {
      __schema {
        queryType {
          fields {
            name
          }
        }
        mutationType {
          fields {
            name
          }
        }
      }
      inputType: __type(name: "Resources_dashboardTopoff_insert_input") {
        name
      }
    }
    """

    result = mat.graphQL.execute(operation=query, variables={})

    if result.get("errors"):
        job.log.warning(f"No se pudo validar schema GraphQL: {result.get('errors')}")
        return False

    data = result.get("data") or {}
    schema = data.get("__schema") or {}

    query_fields = [
        field["name"]
        for field in schema.get("queryType", {}).get("fields", [])
    ]

    mutation_fields = [
        field["name"]
        for field in schema.get("mutationType", {}).get("fields", [])
    ]

    if "Resources_dashboardTopoff" not in query_fields:
        job.log.warning("No existe Resources_dashboardTopoff en query_root.")
        return False

    if "insert_Resources_dashboardTopoff" not in mutation_fields:
        job.log.warning("No existe insert_Resources_dashboardTopoff en mutation_root.")
        return False

    if not data.get("inputType"):
        job.log.warning("No existe Resources_dashboardTopoff_insert_input.")
        return False

    return True
    
## END OF HELPERS ##

def download_sftp_files(**kwargs):
    # Obtener credenciales desde inventario
    hostname, username, password = get_sftp_credentials()
    remote_path = '/'
    # Local path
    current_dir = get_work_base_dir(kwargs)
    print(current_dir)
    os.makedirs(f'{current_dir}/sftp_files', exist_ok=True)
    os.makedirs(f'{current_dir}/Procesados', exist_ok=True)
    local_path = os.path.join(current_dir, "sftp_files")
    print(f"local_path: {local_path}")
    permisos = 0o777
    os.chmod(local_path, permisos) 
    
    
    # String a buscar en el nombre del archivo
    search_string = '_TOPOFF.zip'
    
    # Conexión servidor SFTP usando SSHClient
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh_client.connect(
            hostname=hostname,
            username=username,
            password=password
        )

        sftp_client = ssh_client.open_sftp()
        print("Conectado al servidor SFTP.")

        last_hour = datetime.now() - timedelta(hours=4)
        date_hour_pattern = last_hour.strftime("%Y%m%d")

        print(f"date_hour_pattern: {date_hour_pattern}")
        print(f"Buscando archivos con el patrón de fecha: {date_hour_pattern}")

        files_to_download = []

        for filename in sftp_client.listdir(remote_path):
            if date_hour_pattern in filename and search_string in filename:
                files_to_download.append(filename)

        if not files_to_download:
            print(
                f"No se encontraron archivos con el patrón "
                f"'{date_hour_pattern}' y '{search_string}'."
            )
            raise AirflowSkipException("No hay archivos disponibles para esta ventana de ejecucion.")

        print(f"Se encontraron {len(files_to_download)} archivos para descargar.")

        for filename in files_to_download:
            remote_file_path = os.path.join(remote_path, filename)
            local_file_path = os.path.join(local_path, filename)

            print(f"Downloading '{filename}' to '{local_file_path}'...")
            sftp_client.get(remote_file_path, local_file_path)
            print("Download successful!")

    except paramiko.AuthenticationException:
        print("Authentication failed. Check your credentials.")
        raise

    except paramiko.SSHException as e:
        print(f"SSH connection error: {e}")
        raise

    except FileNotFoundError:
        print(f"Remote path '{remote_path}' or local path '{local_path}' not found.")
        raise

    finally:
        if "sftp_client" in locals():
            sftp_client.close()

        if "ssh_client" in locals():
            ssh_client.close()

        print("SFTP and SSH connections closed.")
        
def extraction_batch(**kwargs):
    current_dir = get_work_base_dir(kwargs)
    #Crea carpeta 'extracted_files' para extrar los csv's resultantes
    os.makedirs(os.path.join(current_dir, "extracted_files"), exist_ok=True)
    #Paths de Origen y destino de los archivos comprimidos / descomprimidos
    source_dir = os.path.join(current_dir, "sftp_files")
    dest_dir = os.path.join(current_dir, 'extracted_files')
    #Iteracion de rutina para descomprimir los archivos de ruta fuente a ruta destino
    for filename in os.listdir(source_dir):
        if filename.endswith(".zip"):
            gz_path = os.path.join(source_dir, filename)
            output_path = os.path.join(dest_dir, filename[:-4])  # Remove .zip

#            with zip.ZipFile(gz_path, 'r') as f_in, open(output_path, 'wb') as f_out:
#                shutil.copyfileobj(f_in, f_out)

            with zip.ZipFile(gz_path, 'r') as f_out:
                f_out.extractall(dest_dir)

                print(f"Descomprimido: {filename} --> {os.path.basename(output_path)}")

            
    
#    /usr/local/iquall/mat/shared/sandbox/apps/9jd-guh-r7j/extracted_files

def df_ericsson4g(**kwargs):
    current_dir = get_work_base_dir(kwargs)
    filepath = os.path.join(current_dir, 'extracted_files/DASH_4G_E_*.csv')
    files = glob.glob(filepath)
    if not files:
        print('No se encontraron archivos')
    else:    
        ####Carga y procesa los top offenders para 4G ERICSSON####
        # Define el directorio y patron de busqueda
        directory = os.path.join(current_dir, 'extracted_files')
        pattern = 'DASH_4G_E_*.csv'  # Patron de busqueda
        
        # Directorio de busqueda
        search_path = os.path.join(directory, pattern)
        
        #Busca todos los archivos que hagan match
        matching_files = glob.glob(search_path)
        
        # Carga y junta todos los archivos que hacen match
        dataframes = [pd.read_csv(file) for file in matching_files]
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        split_name = pattern.split('_')
        Tech = split_name[1] + " " + split_name[2]
        print(Tech)
        
        # === Renombrado de columnas ===
        merged_df = merged_df.assign(Tech= Tech)
        
        df_complete = merged_df.rename(columns={
        'TRAFFIC_PS_TOTAL_GB': 'PS_TRAFF_GB'
        ,'RRC_FAILURES_RATE': 'PS_RRC_%IA'
        ,'RRC_FAILURES': 'PS_RRC_FAIL'
        ,'ERAB_FAILURES_RATE': 'PS_RAB_%IA'
        ,'ERAB_FAILURES': 'PS_RAB_FAIL'
        ,'S1_FAILURES_RATE': 'PS_S1_%IA'
        ,'S1_FAILURES': 'PS_S1_FAIL'
        ,'DCR': 'PS_DROP_%DC'
        ,'ERAB_DROPS': 'PS_DROP_ABNREL'
        ,'USER_TRAFFIC_VOLTE': 'CS_TRAFF_ERL'
        ,'FAILURES_ACC_VOLTE': 'CS_RAB_%IA'
        ,'VOLTE_DCR': 'CS_DROP_%DC'
        ,'MIN_UNAVAILTIME': 'Unav'
        ,'ENB_AGG': 'RNC'
        ,'ENODEB': 'NODEB'
        })
        
        df_complete = df_complete.drop_duplicates()
        print(f"Ruta donde se guardara el archivo: {os.path.join(current_dir, 'Procesados')}" )
        df_complete.to_csv(os.path.join(current_dir, 'Procesados/4g_ericsson.csv'), index=False)
        print('Creacion correcta del csv')
        del df_complete
        
def df_samsung4g(**kwargs):
    current_dir = get_work_base_dir(kwargs)
    filepath = os.path.join(current_dir, 'extracted_files/DASH_4G_S_*.csv')
    files = glob.glob(filepath)
    if not files:
        print('No se encontraron archivos')
    else:    
        ####Carga y procesa los top offenders para 4G SAMSUNG####
        # Define el directorio y patron de busqueda
        directory = os.path.join(current_dir, 'extracted_files')
        pattern = 'DASH_4G_S_*.csv'  # Patron de busqueda
        
        # Directorio de busqueda
        search_path = os.path.join(directory, pattern)
        
        #Busca todos los archivos que hagan match
        matching_files = glob.glob(search_path)
        
        # Carga y junta todos los archivos que hacen match
        dataframes = [pd.read_csv(file) for file in matching_files]
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        split_name = pattern.split('_')
        Tech = split_name[1] + " " + split_name[2]
        print(Tech)
        
        # === Renombrado de columnas ===
        merged_df = merged_df.assign(Tech= Tech)
        
        df_complete = merged_df.rename(columns={
        'TRAFFIC_D_USER_PS_GB': 'PS_TRAFF_GB'
        ,'RRC_FAILURES_RATE': 'PS_RRC_%IA'
        ,'RRC_FAILURES': 'PS_RRC_FAIL'
        ,'ERAB_FAILURES_RATE': 'PS_RAB_%IA'
        ,'ERAB_FAILURES': 'PS_RAB_FAIL'
        ,'S1_FAILURES_RATE': 'PS_S1_%IA'
        ,'S1_FAILURES': 'PS_S1_FAIL'
        ,'DCR': 'PS_DROP_%DC'
        ,'ERAB_DROPS': 'PS_DROP_ABNREL'
        ,'USER_TRAFFIC_VOLTE': 'CS_TRAFF_ERL'
        ,'FAILURES_ACC_VOLTE': 'CS_RAB_%IA'
        ,'VOLTE_DCR': 'CS_DROP_%DC'
        ,'MIN_UNAVAILTIME': 'Unav'
        })
        
        df_complete = df_complete.drop_duplicates()
        print(f"Ruta donde se guardara el archivo: {os.path.join(current_dir, 'Procesados')}" )
        df_complete.to_csv(os.path.join(current_dir, 'Procesados/4g_samsung.csv'), index=False)
        print('Creacion correcta del csv')
        del df_complete
        
def df_huawei4g(**kwargs):
    current_dir = get_work_base_dir(kwargs)
    filepath = os.path.join(current_dir, 'extracted_files/DASH_4G_H_*.csv')
    files = glob.glob(filepath)
    if not files:
        print('No se encontraron archivos')
    else:    
        ####Carga y procesa los top offenders para 4G HUAWEI####
        # Define el directorio y patron de busqueda
        directory = os.path.join(current_dir, 'extracted_files')
        pattern = 'DASH_4G_H_*.csv'  # Patron de busqueda
        
        # Directorio de busqueda
        search_path = os.path.join(directory, pattern)
        
        #Busca todos los archivos que hagan match
        matching_files = glob.glob(search_path)
        
        # Carga y junta todos los archivos que hacen match
        dataframes = [pd.read_csv(file) for file in matching_files]
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        split_name = pattern.split('_')
        Tech = split_name[1] + " " + split_name[2]
        print(Tech)
        
        # === Renombrado de columnas ===
        merged_df = merged_df.assign(Tech= Tech)
        
        df_complete = merged_df.rename(columns={
         'TRAFFIC_PS_TOTAL_GB': 'PS_TRAFF_GB'
        ,'RRC_FAILURES_RATE': 'PS_RRC_%IA'
        ,'RRC_FAILURES': 'PS_RRC_FAIL'
        ,'ERAB_FAILURES_RATE': 'PS_RAB_%IA'
        ,'ERAB_FAILURES': 'PS_RAB_FAIL'
        ,'S1_FAILURES_RATE': 'PS_S1_%IA'
        ,'S1_FAILURES': 'PS_S1_FAIL'
        ,'DCR': 'PS_DROP_%DC'
        ,'ERAB_DROPS': 'PS_DROP_ABNREL'
        ,'USER_TRAFFIC_VOLTE': 'CS_TRAFF_ERL'
        ,'FAILURES_ACC_VOLTE': 'CS_RAB_%IA'
        ,'VOLTE_DCR': 'CS_DROP_%DC'
        ,'MIN_UNAVAILTIME': 'Unav'
        ,'CONT_ABNORMREL_TNL_RATE': '3G_RTX/4G_TNL_%Tx'
        ,'L_ERAB_ABNORMREL_TNL': 'TNL_ABN'
        ,'TNL_FAILEST': 'TNL_FAIL'
        })
        
        df_complete = df_complete.drop_duplicates()
        print(f"Ruta donde se guardara el archivo: {os.path.join(current_dir, 'Procesados')}" )
        df_complete.to_csv(os.path.join(current_dir, 'Procesados/4g_huawei.csv'), index=False)
        print('Creacion correcta del csv')
        del df_complete

def df_nokia4g(**kwargs):
    current_dir = get_work_base_dir(kwargs)
    filepath = os.path.join(current_dir, 'extracted_files/DASH_4G_N_*.csv')
    files = glob.glob(filepath)
    if not files:
        print('No se encontraron archivos')
    else:    
        ####Carga y procesa los top offenders para 4G NOKIA####
        # Define el directorio y patron de busqueda
        directory = os.path.join(current_dir, 'extracted_files')
        pattern = 'DASH_4G_N_*.csv'  # Patron de busqueda
        
        # Directorio de busqueda
        search_path = os.path.join(directory, pattern)
        
        #Busca todos los archivos que hagan match
        matching_files = glob.glob(search_path)
        
        # Carga y junta todos los archivos que hacen match
        dataframes = [pd.read_csv(file) for file in matching_files]
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        split_name = pattern.split('_')
        Tech = split_name[1] + " " + split_name[2]
        print(Tech)
        
        # === Renombrado de columnas ===
        merged_df = merged_df.assign(Tech= Tech)
        
        df_complete = merged_df.rename(columns={
        'TRAFFIC_D_USER_PS_GB': 'PS_TRAFF_GB'
        ,'RRC_FAILURES_RATE': 'PS_RRC_%IA'
        ,'RRC_FAILURES': 'PS_RRC_FAIL'
        ,'ERAB_FAILURES_RATE': 'PS_RAB_%IA'
        ,'ERAB_FAILURES': 'PS_RAB_FAIL'
        ,'S1_FAILURES_RATE': 'PS_S1_%IA'
        ,'S1_FAILURES': 'PS_S1_FAIL'
        ,'DCR': 'PS_DROP_%DC'
        ,'ERAB_DROPS': 'PS_DROP_ABNREL'
        ,'USER_TRAFFIC_VOLTE': 'CS_TRAFF_ERL'
        ,'FAILURES_ACC_VOLTE': 'CS_RAB_%IA'
        ,'VOLTE_DCR': 'CS_DROP_%DC'
        ,'MIN_UNAVAILTIME': 'Unav'
        })
        
        df_complete = df_complete.drop_duplicates()
        print(f"Ruta donde se guardara el archivo: {os.path.join(current_dir, 'Procesados')}" )
        df_complete.to_csv(os.path.join(current_dir, 'Procesados/4g_nokia.csv'), index=False)
        print('Creacion correcta del csv')
        del df_complete

        
def df_nokia3g(**kwargs):
    current_dir = get_work_base_dir(kwargs)
    filepath = os.path.join(current_dir, 'extracted_files/DASH_3G_N_*.xlsx')
    files = glob.glob(filepath)
    if not files:
        print('No se encontraron archivos')
    else:    
        ####Carga y procesa los top offenders para 3G NOKIA####
        # Define el directorio y patron de busqueda
        directory = os.path.join(current_dir, 'extracted_files')
        pattern = 'DASH_3G_N_*.xlsx'  # Patron de busqueda
        
        # Directorio de busqueda
        search_path = os.path.join(directory, pattern)
        
        #Busca todos los archivos que hagan match
        matching_files = glob.glob(search_path)
        
        # Carga y junta todos los archivos que hacen match
        #dataframes = [pd.read_excel(file, skiprows = 1) for file in matching_files]
        dataframes = [pd.read_excel(file,engine='openpyxl', skiprows=1) for file in matching_files]
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        split_name = pattern.split('_')
        Tech = split_name[1] + " " + split_name[2]
        print(Tech)
        
        # === Renombrado de columnas ===
        merged_df = merged_df.assign(Tech= Tech)
        
        df_complete = merged_df.rename(columns={
        'TRAFFIC_TOTAL_GB': 'PS_TRAFF_GB'
        ,'RRC_FAILURES_RATE_PS': 'PS_RRC_%IA'
        ,'PS_FAILURE_RRC': 'PS_RRC_FAIL'
        ,'RAB_FAILURES_RATE_PS': 'PS_RAB_%IA'
        ,'PS_FAILURES_RAB': 'PS_RAB_FAIL'
        ,'PS_DCR': 'PS_DROP_%DC'
        ,'PS_RETAINABILITY_NUM': 'PS_DROP_ABNREL'
        ,'TRAFFIC_CS': 'CS_TRAFF_ERL'
        ,'RRC_FAILURES_RATE_CS': 'CS_RRC_%IA'
        ,'CS_FAILURES_RRC': 'CS_RRC_FAIL'
        ,'RAB_FAILURES_RATE_CS': 'CS_RAB_%IA'
        ,'CS_FAILURES_RAB': 'CS_RAB_FAIL'
        ,'VOICE_DCR': 'CS_DROP_%DC'
        ,'DROPS_VOICE': 'CS_DROP_ABNREL'
        ,'MIN_UNAVAILTIME': 'Unav'
        })
        
        df_complete = df_complete.drop_duplicates()
        print(f"Ruta donde se guardara el archivo: {os.path.join(current_dir, 'Procesados')}" )
        df_complete.to_csv(os.path.join(current_dir, 'Procesados/3g_nokia.csv'), index=False)
        print('Creacion correcta del csv')
        del df_complete
        
def df_huawei3g(**kwargs):
    current_dir = get_work_base_dir(kwargs)
    filepath = os.path.join(current_dir, 'extracted_files/DASH_3G_H_*.csv')
    files = glob.glob(filepath)
    if not files:
        print('No se encontraron archivos')
    else:    
        ####Carga y procesa los top offenders para 3G HUAWEI####
        # Define el directorio y patron de busqueda
        directory = os.path.join(current_dir, 'extracted_files')
        pattern = 'DASH_3G_H_*.csv'  # Patron de busqueda
        
        # Directorio de busqueda
        search_path = os.path.join(directory, pattern)
        
        #Busca todos los archivos que hagan match
        matching_files = glob.glob(search_path)
        
        # Carga y junta todos los archivos que hacen match
        dataframes = [pd.read_csv(file) for file in matching_files]
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        split_name = pattern.split('_')
        Tech = split_name[1] + " " + split_name[2]
        print(Tech)
        
        # === Renombrado de columnas ===
        merged_df = merged_df.assign(Tech= Tech)
        
        df_complete = merged_df.rename(columns={
         'TRAFFIC_TOTAL_GB': 'PS_TRAFF_GB'
        ,'RRC_FAILURES_RATE_PS': 'PS_RRC_%IA'
        ,'PS_FAIILURE_RRC': 'PS_RRC_FAIL'
        ,'RAB_FAILURES_RATE_PS': 'PS_RAB_%IA'
        ,'PS_FAILURES_RAB': 'PS_RAB_FAIL'
        ,'LCS_PS_RATE': 'PS_DROP_%DC'
        ,'PS_ABNORMAL_RELEASES': 'PS_DROP_ABNREL'
        ,'TRAFFIC_V_USER_CS': 'CS_TRAFF_ERL'
        ,'CS_FAILURE_RRC_RATE': 'CS_RRC_%IA'
        ,'CS_FAILURES_RRC': 'CS_RRC_FAIL'
        ,'CS_FAILURE_RAB_RATE': 'CS_RAB_%IA'
        ,'CS_FAILURES_RAB': 'CS_RAB_FAIL'
        ,'LCS_CS_RATE': 'CS_DROP_%DC'
        ,'CS_ABNORMAL_RELEASES': 'CS_DROP_ABNREL'
        ,'MIN_UNAVAILTIME': 'Unav'
        ,'RETRANSMISSION_RATE': '3G_RTX/4G_TNL_%Tx'
        })
        
        df_complete = df_complete.drop_duplicates()
        print(f"Ruta donde se guardara el archivo: {os.path.join(current_dir, 'Procesados')}" )
        df_complete.to_csv(os.path.join(current_dir, 'Procesados/3g_huawei.csv'), index=False)
        print('Creacion correcta del csv')
        del df_complete
        
def df_ericsson3g(**kwargs):
    current_dir = get_work_base_dir(kwargs)
    filepath = os.path.join(current_dir, 'extracted_files/DASH_3G_E_*.csv')
    files = glob.glob(filepath)
    if not files:
        print('No se encontraron archivos')
    else:    
        ####Carga y procesa los top offenders para 3G ERICSSON####
        # Define el directorio y patron de busqueda
        directory = os.path.join(current_dir, 'extracted_files')
        pattern = 'DASH_3G_E_*.csv'  # Patron de busqueda
        
        # Directorio de busqueda
        search_path = os.path.join(directory, pattern)
        
        #Busca todos los archivos que hagan match
        matching_files = glob.glob(search_path)
        
        # Carga y junta todos los archivos que hacen match
        dataframes = [pd.read_csv(file) for file in matching_files]
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        split_name = pattern.split('_')
        Tech = split_name[1] + " " + split_name[2]
        print(Tech)
        
        # === Renombrado de columnas ===
        merged_df = merged_df.assign(Tech= Tech)
        
        df_complete = merged_df.rename(columns={
         'TRAFFIC_TOTAL_GB': 'PS_TRAFF_GB'
        ,'RRC_FAILURES_RATE_PS': 'PS_RRC_%IA'
        ,'PS_FAIILURE_RRC': 'PS_RRC_FAIL'
        ,'RAB_FAILURES_RATE_PS': 'PS_RAB_%IA'
        ,'PS_FAILURES_RAB': 'PS_RAB_FAIL'
        ,'PS_DCR': 'PS_DROP_%DC'
        ,'PS_ABNORMAL_RELEASES': 'PS_DROP_ABNREL'
        ,'TRAFFIC_CS_NOCPERF': 'CS_TRAFF_ERL'
        ,'RRC_FAILURES_RATE_CS': 'CS_RRC_%IA'
        ,'RRC_FAILURES_CS': 'CS_RRC_FAIL'
        ,'RAB_FAILURES_RATE_CS': 'CS_RAB_%IA'
        ,'CS_FAILURES_RAB': 'CS_RAB_FAIL'
        ,'VOICE_DCR': 'CS_DROP_%DC'
        ,'DROPS_VOICE': 'CS_DROP_ABNREL'
        ,'MIN_UNAVAILTIME': 'Unav'
        })
        
        df_complete = df_complete.drop_duplicates()
        print(f"Ruta donde se guardara el archivo: {os.path.join(current_dir, 'Procesados')}" )
        df_complete.to_csv(os.path.join(current_dir, 'Procesados/3g_ericsson.csv'), index=False)
        print('Creacion correcta del csv')
        del df_complete
    
def df_merged(**kwargs):    
        # Ruta donde están los archivos CSV
#    folder_path = '/usr/local/iquall/mat/shared/sandbox/apps/9jd-guh-r7j/Procesados'
    current_dir = get_work_base_dir(kwargs)
    folder_path = os.path.join(current_dir, 'Procesados')
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        raise AirflowException(f"No se encontraron archivos para homologar en: {folder_path}")
    else:
        # Lista para almacenar los DataFrames individuales
        dataframes = []
        
        # Leer cada archivo y agregarlo a la lista
        for file in csv_files:
            df = pd.read_csv(file)
            df['Archivo_Fuente'] = os.path.basename(file)  # Agrega columna con el nombre del archivo
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            df['Fecha_Ejecucion'] = timestamp
        
            dataframes.append(df)
        
        # Combinar todos los DataFrames en uno solo
        combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
        cols_move = ['Archivo_Fuente', 'Fecha_Ejecucion']
        cols_move = [col for col in cols_move if col in combined_df.columns]
        nuevo_orden = [col for col in combined_df.columns if col not in cols_move] + cols_move
        os.makedirs(os.path.join(current_dir, 'DF_Consolidado'), exist_ok=True)
        combined_df = combined_df[nuevo_orden]
        combined_df = combined_df.round(2)
        
        #Elimina duplicados
        combined_df = combined_df.drop_duplicates()
        ##Separar vendor y tecnologia
        combined_df[['Technology', 'Vendor']] = combined_df['Tech'].str.split(' ', n=1, expand=True)
        
        ##Mapeao vendor
        size_map = {
            'N': 'NOKIA',
            'H': 'HUAWEI',
            'E': 'ERICSSON',
            'S': 'SAMSUNG'
        }
        
        ##ASignacion de mapeo
        combined_df['Vendor'] = combined_df['Vendor'].map(size_map)
    
        #Reordenado de columnas
        combined_df =  combined_df[[
        'Tech'
        ,'Technology'
        ,'Vendor'
        ,'DATE'
        ,'TIME'
        ,'REGION'
        ,'PROVINCE'
        ,'MUNICIPALITY'
        ,'SITE_ATT'
        ,'RNC'
        ,'NODEB'
        ,'NOC_CLUSTER'
        ,'PS_TRAFF_GB'
        ,'PS_RRC_%IA'
        ,'PS_RRC_FAIL'
        ,'PS_RAB_%IA'
        ,'PS_RAB_FAIL'
        ,'PS_S1_%IA'
        ,'PS_S1_FAIL'
        ,'PS_DROP_%DC'
        ,'PS_DROP_ABNREL'
        ,'CS_TRAFF_ERL'
        ,'CS_RRC_%IA'
        ,'CS_RRC_FAIL'
        ,'CS_RAB_%IA'
        ,'CS_RAB_FAIL'
        ,'CS_DROP_%DC'
        ,'CS_DROP_ABNREL'
        ,'Unav'
        ,'3G_RTX/4G_TNL_%Tx'
        ,'TNL_ABN'
        ,'TNL_FAIL'
        ,'Archivo_Fuente'
        ,'Fecha_Ejecucion'
         ]]
        
        combined_df = combined_df[(combined_df['PROVINCE'] != '') & combined_df['PROVINCE'].notna()]
        print("Resumen TOP_resultante por Archivo_Fuente:")
        print(combined_df.groupby("Archivo_Fuente").size())

        print("Resumen TOP_resultante por Technology/Vendor:")
        print(combined_df.groupby(["Technology", "Vendor"]).size())

        print(f"Total filas TOP_resultante: {len(combined_df)}")
        combined_df.to_csv(os.path.join(current_dir, 'DF_Consolidado/TOP_resultante.csv'), index=False)

def replace_dashboard_topoff_graphql(**kwargs):
    job = kwargs.get("job")
    mat = MATClient()

    if not is_dashboard_topoff_available(mat, job):
        job.log.warning(
            "Recurso dashboardTopoff no disponible en esta ejecucion. "
            "Se omite carga GraphQL y continua el flujo para enviar correo."
        )
        return

    current_dir = get_work_base_dir(kwargs)
    file_path = os.path.join(current_dir, "DF_Consolidado", "TOP_resultante.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No existe el archivo consolidado: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise AirflowException("El archivo consolidado esta vacio. No hay registros para insertar.")

    column_map = {
        "DATE": "Date",
        "TIME": "Time",
        "REGION": "Region",
        "PROVINCE": "Province",
        "MUNICIPALITY": "Municipality",
        "SITE_ATT": "Site_att",
        "NODEB": "NodeB",
        "NOC_CLUSTER": "Noc_Cluster",
        "3G_RTX/4G_TNL_%Tx": "G_RTX4G_TNL__Tx",
        "PS_RRC_%IA": "PS_RRC__IA",
        "PS_RAB_%IA": "PS_RAB__IA",
        "PS_S1_%IA": "PS_S1__IA",
        "PS_DROP_%DC": "PS_DROP__DC",
        "CS_RRC_%IA": "CS_RRC__IA",
        "CS_RAB_%IA": "CS_RAB__IA",
        "CS_DROP_%DC": "CS_DROP__DC",
    }

    df = df.rename(columns=column_map)

    job.log.info(f"Filas leidas de TOP_resultante.csv: {len(df)}")

    if "Archivo_Fuente" in df.columns:
        job.log.info("Resumen por Archivo_Fuente:")
        job.log.info(df.groupby("Archivo_Fuente").size().to_string())

    if "Technology" in df.columns and "Vendor" in df.columns:
        job.log.info("Resumen por Technology/Vendor:")
        job.log.info(df.groupby(["Technology", "Vendor"]).size().to_string())

    required_cols = ["Date", "Time", "Technology", "Vendor"]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise Exception(f"Faltan columnas requeridas para dashboardTopoff: {missing_cols}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date.astype(str)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce").dt.strftime("%H:%M")
    df["Technology"] = df["Technology"].astype(str).str.strip()
    df["Vendor"] = df["Vendor"].astype(str).str.strip()

    key_cols = [
        "Technology",
        "Vendor",
        "Date",
        "Time",
        "Site_att",
        "RNC",
        "NodeB",
        "Noc_Cluster",
    ]

    for col in key_cols:
        if col not in df.columns:
            df[col] = None

    def build_pk_from_values(values):
        return "||".join("" if pd.isna(value) else str(value).strip() for value in values)

    df["__pk"] = df.apply(
        lambda row: build_pk_from_values([row[col] for col in key_cols]),
        axis=1
    )

    before_internal = len(df)
    df = df.drop_duplicates(subset=["__pk"], keep="last").copy()
    after_internal = len(df)

    job.log.info(f"Registros antes de deduplicar CSV por llave compuesta: {before_internal}")
    job.log.info(f"Registros despues de deduplicar CSV por llave compuesta: {after_internal}")

    dates = sorted(df["Date"].dropna().astype(str).unique().tolist())
    times = sorted(df["Time"].dropna().astype(str).unique().tolist())

    if not dates or not times:
        raise Exception("No hay valores validos de Date/Time para consultar existentes en GraphQL.")

    query_existing = """
    query getExistingTopoff($dates: [String!], $times: [String!], $limit: Int!, $offset: Int!) {
      Resources_dashboardTopoff(
        where: {
          Date: { _in: $dates },
          Time: { _in: $times }
        },
        limit: $limit,
        offset: $offset
      ) {
        Technology
        Vendor
        Date
        Time
        Site_att
        RNC
        NodeB
        Noc_Cluster
      }
    }
    """

    existing_keys = set()
    limit = 1000
    offset = 0

    while True:
        result_existing = mat.graphQL.execute(
            operation=query_existing,
            variables={
                "dates": dates,
                "times": times,
                "limit": limit,
                "offset": offset
            }
        )

        if result_existing.get("errors"):
            raise Exception(f"Error consultando registros existentes en GraphQL: {result_existing.get('errors')}")

        existing_rows = (
            result_existing
            .get("data", {})
            .get("Resources_dashboardTopoff", [])
        )

        if not existing_rows:
            break

        for existing_row in existing_rows:
            existing_keys.add(
                build_pk_from_values([existing_row.get(col) for col in key_cols])
            )

        offset += limit

    before_existing_filter = len(df)

    df = df[~df["__pk"].isin(existing_keys)].copy()

    after_existing_filter = len(df)

    job.log.info(f"Registros antes de filtrar existentes en GraphQL: {before_existing_filter}")
    job.log.info(f"Registros nuevos para insertar: {after_existing_filter}")
    job.log.info(f"Registros omitidos porque ya existian: {before_existing_filter - after_existing_filter}")

    df = df.drop(columns=["__pk"])

    if df.empty:
        job.log.info("Todos los registros del CSV ya existen en GraphQL. No hay registros nuevos para insertar.")
        return

    schema_input_query = """
    query getInputType($typeName: String!) {
      __type(name: $typeName) {
        inputFields {
          name
        }
      }
    }
    """

    schema_input_result = mat.graphQL.execute(
        operation=schema_input_query,
        variables={"typeName": "Resources_dashboardTopoff_insert_input"}
    )

    if schema_input_result.get("errors"):
        raise Exception(f"Error consultando schema GraphQL: {schema_input_result.get('errors')}")

    input_fields = (
        schema_input_result
        .get("data", {})
        .get("__type", {})
        .get("inputFields", [])
    )

    graphql_fields = [field["name"] for field in input_fields]

    allowed_cols = [
        "Archivo_Fuente",
        "CS_DROP_ABNREL",
        "CS_DROP__DC",
        "CS_RAB_FAIL",
        "CS_RAB__IA",
        "CS_RRC_FAIL",
        "CS_RRC__IA",
        "CS_TRAFF_ERL",
        "Date",
        "Fecha_Ejecucion",
        "G_RTX4G_TNL__Tx",
        "Municipality",
        "Noc_Cluster",
        "NodeB",
        "PS_DROP_ABNREL",
        "PS_DROP__DC",
        "PS_RAB_FAIL",
        "PS_RAB__IA",
        "PS_RRC_FAIL",
        "PS_RRC__IA",
        "PS_S1_FAIL",
        "PS_S1__IA",
        "PS_TRAFF_GB",
        "Province",
        "RNC",
        "Region",
        "Site_att",
        "TNL_ABN",
        "TNL_FAIL",
        "Technology",
        "Time",
        "Unav",
        "Vendor",
    ]

    available_cols = [
        col for col in allowed_cols
        if col in df.columns and col in graphql_fields
    ]

    df_insert = df[available_cols].copy()

    if df_insert.empty:
        raise Exception("No quedaron columnas validas para insertar en Resources_dashboardTopoff.")

    records = []

    for row in df_insert.to_dict(orient="records"):
        records.append({
            key: clean_value(value)
            for key, value in row.items()
        })

    if not records:
        raise Exception("No hay registros validos para insertar en Resources_dashboardTopoff.")

    job.log.info(f"Columnas finales enviadas a GraphQL: {available_cols}")
    job.log.info(f"Total registros nuevos preparados para insertar: {len(records)}")

    mutation_insert = """
    mutation insertDashboardTopoff($objects: [Resources_dashboardTopoff_insert_input!]!) {
      insert_Resources_dashboardTopoff(objects: $objects) {
        affected_rows
      }
    }
    """

    batch_size = 500
    total_inserted = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]

        result = mat.graphQL.execute(
            operation=mutation_insert,
            variables={"objects": batch}
        )

        job.log.info(f"Resultado insert batch {i} - {i + len(batch)}:")
        job.log.info(json.dumps(result, indent=2))

        if result.get("errors"):
            raise Exception(
                f"Error insertando batch {i} - {i + len(batch)}: {result.get('errors')}"
            )

        total_inserted += (
            result
            .get("data", {})
            .get("insert_Resources_dashboardTopoff", {})
            .get("affected_rows", 0)
        )

    job.log.info(f"Total insertado en Resources_dashboardTopoff: {total_inserted}")
    
def send_notification(**kwargs):
    current_dir = get_work_base_dir(kwargs)

    def _parse_emails(value):
        emails = []

        if value is None:
            return emails

        if isinstance(value, str):
            parts = re.split(r"[,\n;]+", value)
            emails.extend([x.strip() for x in parts if x and x.strip()])

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    parts = re.split(r"[,\n;]+", item)
                    emails.extend([x.strip() for x in parts if x and x.strip()])
                elif isinstance(item, dict):
                    possible = (
                        item.get("email")
                        or item.get("value")
                        or item.get("label")
                        or ""
                    )
                    parts = re.split(r"[,\n;]+", str(possible))
                    emails.extend([x.strip() for x in parts if x and x.strip()])
                else:
                    parts = re.split(r"[,\n;]+", str(item))
                    emails.extend([x.strip() for x in parts if x and x.strip()])

        else:
            parts = re.split(r"[,\n;]+", str(value))
            emails.extend([x.strip() for x in parts if x and x.strip()])

        unique = []
        seen = set()

        for email in emails:
            key = email.lower()
            if key not in seen:
                seen.add(key)
                unique.append(email)

        return unique

    params = getForm(kwargs) or {}
    data = params.get("data") or {}
    ui_email_value = data.get("emailToSend", "")

    dest_ui = _parse_emails(ui_email_value)
    hard_dest = ["salvador.caracoza@innovasolutions.com"]
    dest = dest_ui if dest_ui else hard_dest

    folder_path = os.path.join(current_dir, "Procesados")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print("No se encontraron archivos para Enviar")
        return

    if not dest:
        print("No hay destinatarios configurados")
        return

    last_hour = datetime.now() - timedelta(hours=3)
    date_hour_pattern = last_hour.strftime("%Y%m%d_%H")

    subj = f"Archivo Homologado - Dashboard Master {date_hour_pattern}"
    msg = (
        "Archivo Homologado correspondinete a los diferentes vendors y tecnologias "
        f"correspondiente a la corrida {date_hour_pattern}"
    )
    files = os.path.join(current_dir, "DF_Consolidado/TOP_resultante.csv")

    if not os.path.exists(files):
        raise FileNotFoundError(f"No existe el archivo para enviar: {files}")

    print(f"Enviando correo a: {dest}")
    print(f"Archivo adjunto: {files}")

    ATTSendEmail(
        dest=dest,
        subj=subj,
        msg=msg,
        files=files
    )

    shutil.rmtree(current_dir, ignore_errors=True)

    # Limpiar directorios de descargas y extraccion
    # shutil.rmtree(os.path.join(current_dir, "sftp_files"))
    # shutil.rmtree(os.path.join(current_dir, "extracted_files"))
    # shutil.rmtree(os.path.join(current_dir, "Procesados"))
    
########################################################################################################################
###############################################                     ####################################################
###############################################  TASKS DEFINITIONS  ####################################################
###############################################                     ####################################################
########################################################################################################################


init = MATInstanceInitOperator(task_id='MAT_Inicializacion', dag=dag)
download_sftp_files = MATPythonOperator(task_id="descarga_archivos_sftp", python_callable = download_sftp_files, retries=0,  dag=dag)
extract_sftp_files = MATPythonOperator(task_id="extraccion_archivos__sftp", python_callable = extraction_batch, dag=dag)
df_ericsson3g = MATPythonOperator(task_id="df_ericsson3g", python_callable = df_ericsson3g, dag=dag)
df_ericsson4g = MATPythonOperator(task_id="df_ericsson4g", python_callable = df_ericsson4g, dag=dag)
df_huawei3g = MATPythonOperator(task_id="df_huawei3g", python_callable = df_huawei3g, dag=dag)
df_huawei4g = MATPythonOperator(task_id="df_huawei4g", python_callable = df_huawei4g, dag=dag)
df_nokia3g = MATPythonOperator(task_id="df_nokia3g", python_callable = df_nokia3g, dag=dag)
df_nokia4g = MATPythonOperator(task_id="df_nokia4g", python_callable = df_nokia4g, dag=dag)
df_samsung4g = MATPythonOperator(task_id="df_samsung4g", python_callable = df_samsung4g, dag=dag)
df_consolidado = MATPythonOperator(task_id="consolidacion_final", python_callable = df_merged, dag=dag)
df_insert_graphql = MATPythonOperator(task_id="replace_dashboard_topoff_graphql", python_callable=replace_dashboard_topoff_graphql, dag=dag)
df_send_mail = MATPythonOperator(task_id="enviar_email", python_callable = send_notification, dag=dag)

end = MATInstanceExitOperator(task_id= 'MAT_Finalizar', dag=dag)


########################################################################################################################
################################################                  ######################################################
################################################  TASKS WORKFLOW  ######################################################
################################################                  ######################################################
########################################################################################################################

#init >> download_sftp_files >> extract_sftp_files >> df_ericsson3g >> df_consolidado >> df_send_mail >> end
#>> df_consolidado

init >> download_sftp_files >> extract_sftp_files >> df_ericsson3g >> df_consolidado >> df_insert_graphql >> df_send_mail >> end
extract_sftp_files >> df_ericsson4g >> df_consolidado
extract_sftp_files >> df_nokia3g >> df_consolidado
extract_sftp_files >> df_nokia4g >> df_consolidado
extract_sftp_files >> df_huawei3g >> df_consolidado
extract_sftp_files >> df_huawei4g >> df_consolidado
extract_sftp_files >> df_samsung4g >> df_consolidado