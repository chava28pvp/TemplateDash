# coding: utf-8
########################################################################################################################
###########################################                       ######################################################
###########################################  READ ONLY LIBRARIES  ######################################################
###########################################                       ######################################################
########################################################################################################################
import json, logging, os, sys, glob
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
from pathlib import Path
from airflow.utils.email import send_email
from sendmail import ATTSendEmail
import shutil
import glob
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

dag = DAG(dag_id='9jd-guh-r7j', description='', start_date=datetime(2025,9,17), schedule_interval='*/5 * * * *', catchup=False, on_failure_callback=MATWorkflowErrorManagement, on_success_callback=MATWorkflowSuccessManagement, default_args=default_args)


########################################################################################################################
##############################################                    ######################################################
##############################################  CUSTOM FUNCTIONS  ######################################################
##############################################                    ######################################################
########################################################################################################################

## HELPERS ##
def get_sftp_credentials():
    inv = Inventory()
    cipher = crypto.aes256.MATCipher()

    sftp_data = inv.get(
        module="mathost",
        query="data.hostname=EPT_DM",
    )

    if not sftp_data:
        raise ValueError("No se encontró el host EPT_DM en el inventario.")

    data = sftp_data[0]["data"]

    credentials = data["networkRole"]["data"]["matcredentials"]["data"]

    hostname = data["managementIp"]
    username = credentials["username"]
    password = cipher.decrypt(credentials["password"])

    return hostname, username, password
    
def is_dashboard_master_available(mat, job):
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
      inputType: __type(name: "Resources_dashboardMaster_insert_input") {
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

    if "Resources_dashboardMaster" not in query_fields:
        job.log.warning("No existe Resources_dashboardMaster en query_root.")
        return False

    if "insert_Resources_dashboardMaster" not in mutation_fields:
        job.log.warning("No existe insert_Resources_dashboardMaster en mutation_root.")
        return False

    if not data.get("inputType"):
        job.log.warning("No existe Resources_dashboardMaster_insert_input.")
        return False

    return True

## END OF HELPERS ##
def get_work_base_dir(kwargs):
    """
    Directorio dinamico por corrida.
    Todo lo temporal del DAG vive dentro de esta subcarpeta y se elimina al final.
    """
    base_dir = Path(MATDirectories(kwargs).run) / "dashboard_master_work"
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir)

def download_sftp_files(**kwargs):
    # Parámetros de conexión desde inventario cifrado
    hostname, username, password = get_sftp_credentials()

    remote_path = "/"

    # Local path
    current_dir = get_work_base_dir(kwargs)

    os.makedirs(f"{current_dir}/sftp_files", exist_ok=True)
    os.makedirs(f"{current_dir}/Procesados", exist_ok=True)

    local_path = os.path.join(current_dir, "sftp_files")
    print(f"local_path: {local_path}")

    permisos = 0o777
    os.chmod(local_path, permisos)

    # String a buscar en el nombre del archivo
    search_string = "NOC_CLUSTER"

    # Conexión servidor SFTP usando SSHClient
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Conexión al servidor SFTP
        ssh_client.connect(
            hostname=hostname,
            username=username,
            password=password
        )

        sftp_client = ssh_client.open_sftp()
        print("Conectado al servidor SFTP.")
        
        # Generar el patrón de fecha y hora
        last_hour = datetime.now() - timedelta(hours=3)
        date_hour_pattern = last_hour.strftime("%Y%m%d_%H")

        print(f"date_hour_pattern: {date_hour_pattern}")
        print(f"Buscando archivos con el patrón de fecha y hora: {date_hour_pattern}")

        # Listar archivos y filtrar por patrón de nombre
        files_to_download = []

        for filename in sftp_client.listdir(remote_path):
            if date_hour_pattern in filename and search_string in filename:
                files_to_download.append(filename)

        expected_prefixes = [
            "ATT_ERICSSON3G_NOC_CLUSTER",
            "ATT_ERICSSON4G_NOC_CLUSTER",
            "ATT_HUAWEI3G_NOC_CLUSTER",
            "ATT_HUAWEI4G_NOC_CLUSTER",
            "ATT_NOKIA3G_NOC_CLUSTER",
            "ATT_NOKIA4G_NOC_CLUSTER",
            "ATT_SAMSUNG4G_NOC_CLUSTER",
        ]

        missing_prefixes = [
            prefix for prefix in expected_prefixes
            if not any(
                filename.startswith(prefix)
                and date_hour_pattern in filename
                for filename in files_to_download
            )
        ]

        print("===== VALIDACION ARCHIVOS ESPERADOS =====")
        print(f"date_hour_pattern usado: {date_hour_pattern}")
        print(f"files_to_download: {files_to_download}")
        print(f"total files_to_download: {len(files_to_download)}")
        print(f"missing_prefixes: {missing_prefixes}")
        print("===== FIN VALIDACION ARCHIVOS ESPERADOS =====")

        if missing_prefixes:
            raise AirflowException(
                f"Archivos incompletos para {date_hour_pattern}. "
                f"Faltantes: {missing_prefixes}. Reintentando..."
            )

        print(f"Se encontraron los {len(files_to_download)} archivos esperados para descargar.")

        # Descargar archivos filtrados
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

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    finally:
        # Cerrar todas las conexiones
        if "sftp_client" in locals():
            sftp_client.close()

        if "ssh_client" in locals():
            ssh_client.close()

        print("SFTP and SSH connections closed.")


def extraction_batch(**kwargs):
    # Crea carpeta 'extracted_files' para extrar los csv's resultantes
    current_dir = get_work_base_dir(kwargs)
    os.makedirs(f'{current_dir}/extracted_files', exist_ok=True)
    os.makedirs(f'{current_dir}/Procesados', exist_ok=True)
    # Paths de Origen y destino de los archivos comprimidos / descomprimidos
    source_dir = os.path.join(current_dir, "sftp_files")
    dest_dir = os.path.join(current_dir, 'extracted_files')
    # Iteracion de rutina para descomprimir los archivos de ruta fuente a ruta destino
    for filename in os.listdir(source_dir):
        if filename.endswith(".gz"):
            gz_path = os.path.join(source_dir, filename)
            output_path = os.path.join(dest_dir, filename[:-3])  # Remove .gz

            with gzip.open(gz_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

                print(f"Descomprimido: {filename} --> {os.path.basename(output_path)}")


#    /usr/local/iquall/mat/shared/sandbox/apps/9jd-guh-r7j/extracted_files

def df_ericsson3g(**kwargs):
        current_dir = get_work_base_dir(kwargs)
        pattern = os.path.join(current_dir, 'extracted_files', 'ATT_ERICSSON3G_NOC_CLUSTER_*.csv')
        files = glob.glob(pattern)
        if not files:
            print('No se encontraron archivos')
            return
        latest = max(files, key=os.path.getmtime)
        print(f" Archivo encontrado: {latest}")
        df = pd.read_csv(latest)
        file_name = os.path.basename(latest)

        split_name = file_name.split('_')
        provider = split_name[1][:-2]
        technology = split_name[1][-2:]
        print(provider, " ", technology)

        # === DF for net values ===
        df_net = df.assign(Vendor=provider, Technology=technology, Network='NET')

        filtered_columns_net = df_net[['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA TRAFFIC_D_USER_PS_MB', 'TRAFFIC_D_USER_PS_MB', '% PS_FAIILURE_RRC',
                                       'PS_FAIILURE_RRC', '% PS_FAILURES_RAB', 'PS_FAILURES_RAB', 'PS_DCR',
                                       'PS_ABNORMAL_RELEASES', 'DELTA TRAFFIC_CS_NOCPERF', 'TRAFFIC_CS_NOCPERF',
                                       '% RRC_FAILURES_CS', 'RRC_FAILURES_CS', '% CS_FAILURES_RAB', 'CS_FAILURES_RAB',
                                       'DROPS_VOICE'
                                       ]]

        df_net = filtered_columns_net.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            , 'DELTA TRAFFIC_D_USER_PS_MB': 'PS_TRAFF_DELTA'
            , 'TRAFFIC_D_USER_PS_MB': 'PS_TRAFF_GB'
            , '% PS_FAIILURE_RRC': 'PS_RRC_%IA'
            , 'PS_FAIILURE_RRC': 'PS_RRC_FAIL'
            , '% PS_FAILURES_RAB': 'PS_RAB_%IA'
            , 'PS_FAILURES_RAB': 'PS_RAB_FAIL'
            , 'PS_DCR': 'PS_DROP_%DC'
            , 'PS_ABNORMAL_RELEASES': 'PS_DROP_ABNREL'
            , 'DELTA TRAFFIC_CS_NOCPERF': 'CS_TRAFF_DELTA'
            , 'TRAFFIC_CS_NOCPERF': 'CS_TRAFF_ERL'
            , '% RRC_FAILURES_CS': 'CS_RRC_%IA'
            , 'RRC_FAILURES_CS': 'CS_RRC_FAIL'
            , '% CS_FAILURES_RAB': 'CS_RAB_%IA'
            , 'CS_FAILURES_RAB': 'CS_RAB_FAIL'
            , 'DROPS_VOICE': 'CS_DROP_ABNREL'
        })

        col_mb_gb = ['PS_TRAFF_GB']
        df_net[col_mb_gb] = df_net[col_mb_gb] / 1024

        # === DF for ATT values ===
        df_ATT = df.assign(Vendor=provider, Technology=technology, Network='ATT')
        filtered_columns_ATT = df_ATT[
            ['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY', 'DELTA TRAFFIC_GB_ATT',
             'TRAFFIC_GB_ATT', 'DELTA TRAFFIC_AMR_ATT', 'TRAFFIC_AMR_ATT']]
        # filtered_columns_ATT = df_ATT[['Network', 'Technology','Vendor',  'NOC_CLUSTER','DATE', 'TIME', 'INTEGRITY','DELTA TRAFFIC_GB_ATT'	,'TRAFFIC_GB_ATT'	,'DELTA TRAFFIC_AMR_ATT'	,'TRAFFIC_AMR_ATTT']]

        df_ATT = filtered_columns_ATT.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            , 'DELTA TRAFFIC_GB_ATT': 'PS_TRAFF_DELTA'
            , 'TRAFFIC_GB_ATT': 'PS_TRAFF_GB'
            , 'DELTA TRAFFIC_AMR_ATT': 'CS_TRAFF_DELTA'
            , 'TRAFFIC_AMR_ATT': 'CS_TRAFF_ERL'
        })
        # === DF for TEF values ===
        df_TEF = df.assign(Vendor=provider, Technology=technology, Network='TEF')
        filtered_columns_TEF = df_TEF[
            ['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY', 'DELTA TRAFFIC_GB_PLMN2',
             'TRAFFIC_GB_PLMN2', 'DELTA TRAFFIC_AMR_PLMN2', 'TRAFFIC_AMR_PLMN2']]
        df_TEF = filtered_columns_TEF.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            , 'DELTA TRAFFIC_GB_PLMN2': 'PS_TRAFF_DELTA'
            , 'TRAFFIC_GB_PLMN2': 'PS_TRAFF_GB'
            , 'DELTA TRAFFIC_AMR_PLMN2': 'CS_TRAFF_DELTA'
            , 'TRAFFIC_AMR_PLMN2': 'CS_TRAFF_ERL'
        })

        dataframes = [df_net, df_ATT, df_TEF]

        df_combinado = pd.concat(dataframes, ignore_index=True, sort=False)
        priority_cols = ['Network', 'Technology', 'Vendor']
        df_combinado = df_combinado[priority_cols + [col for col in df_combinado.columns if col not in priority_cols]]
        #os.makedirs('/usr/local/iquall/mat/shared/sandbox/apps/9jd-guh-r7j/Procesados', exist_ok=True)
        #os.makedirs(f'{current_dir}/Procesados', exist_ok=True)

        df_combinado.to_csv(f"{current_dir}/Procesados/df_ericsson3g.csv",
                            index=False)


def df_ericsson4g(**kwargs):
        current_dir = get_work_base_dir(kwargs)
        pattern = os.path.join(current_dir, 'extracted_files', 'ATT_ERICSSON4G_NOC_CLUSTER_*.csv')
        files = glob.glob(pattern)
        if not files:
            print('No se encontraron archivos')
            return
        latest = max(files, key=os.path.getmtime)
        print(f" Archivo encontrado: {latest}")
        df = pd.read_csv(latest)
        file_name = os.path.basename(latest)

        split_name = file_name.split('_')
        provider = split_name[1][:-2]
        technology = split_name[1][-2:]
        print(provider, " ", technology)

        # === DF for net values ===
        df_net = df.assign(Vendor=provider, Technology=technology, Network='NET')

        filtered_columns_net = df_net[['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA TRAFFIC_DL_UL_MB', 'TRAFFIC_DL_UL_MB', '% RRC_FAILURES', 'RRC_FAILURES',
                                       '% ERAB_FAILURES', 'ERAB_FAILURES', '% S1_FAILURES', 'S1_FAILURES',
                                       '% ERAB_DROPS', 'ERAB_DROPS']]

        df_net = filtered_columns_net.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            #   ,'INTEGRITY': 'Integrity'
            , 'DELTA TRAFFIC_DL_UL_MB': 'PS_TRAFF_DELTA'
            , 'TRAFFIC_DL_UL_MB': 'PS_TRAFF_GB'
            , '% RRC_FAILURES': 'PS_RRC_%IA'
            , 'RRC_FAILURES': 'PS_RRC_FAIL'
            , '% ERAB_FAILURES': 'PS_RAB_%IA'
            , 'ERAB_FAILURES': 'PS_RAB_FAIL'
            , '% S1_FAILURES': 'PS_S1_%IA'
            , 'S1_FAILURES': 'PS_S1_FAIL'
            , '% ERAB_DROPS': 'PS_DROP_%DC'
            , 'ERAB_DROPS': 'PS_DROP_ABNREL'
        })

        col_mb_gb = ['PS_TRAFF_GB']
        df_net[col_mb_gb] = df_net[col_mb_gb] / 1024

        # === DF for ATT values ===
        df_ATT = df.assign(Vendor=provider, Technology=technology, Network='ATT',
                           SUM_PLMN_TRAFFIC_GB=df['PLMN_0_TRAFFIC_GB'] + df['PLMN_1_TRAFFIC_GB'])
        filtered_columns_ATT = df_ATT[['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA PLMN_0_TRAFFIC_GB', 'SUM_PLMN_TRAFFIC_GB', 'PLMN_0_DCR'
                                       ]]
        df_ATT = filtered_columns_ATT.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            , 'DELTA PLMN_0_TRAFFIC_GB': 'PS_TRAFF_DELTA'
            , 'SUM_PLMN_TRAFFIC_GB': 'PS_TRAFF_GB'
            , 'PLMN_0_DCR': 'PS_DROP_%DC'
        })
        # === DF for TEF values ===
        df_TEF = df.assign(Vendor=provider, Technology=technology, Network='TEF')
        filtered_columns_TEF = df_TEF[['Network', 'Vendor', 'Technology', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA PLMN_2_TRAFFIC_GB', 'PLMN_2_TRAFFIC_GB', 'PLMN_2_DCR'
                                       ]]
        df_TEF = filtered_columns_TEF.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            , 'DELTA PLMN_2_TRAFFIC_GB': 'PS_TRAFF_DELTA'
            , 'PLMN_2_TRAFFIC_GB': 'PS_TRAFF_GB'
            , 'PLMN_2_DCR': 'PS_DROP_%DC'
        })

        dataframes = [df_net, df_ATT, df_TEF]

        df_combinado = pd.concat(dataframes, ignore_index=True, sort=False)
        priority_cols = ['Network', 'Technology', 'Vendor']
        df_combinado = df_combinado[priority_cols + [col for col in df_combinado.columns if col not in priority_cols]]
        #os.makedirs('/usr/local/iquall/mat/shared/sandbox/apps/9jd-guh-r7j/Procesados', exist_ok=True)

        df_combinado.to_csv(f"{current_dir}/Procesados/df_ericsson4g.csv",
                            index=False)


def df_huawei3g(**kwargs):
        current_dir = get_work_base_dir(kwargs)
        pattern = os.path.join(current_dir, 'extracted_files', 'ATT_HUAWEI3G_NOC_CLUSTER_*.csv')
        files = glob.glob(pattern)
        if not files:
            print('No se encontraron archivos')
            return
        latest = max(files, key=os.path.getmtime)
        print(f" Archivo encontrado: {latest}")
        df = pd.read_csv(latest)
        file_name = os.path.basename(latest)

        split_name = file_name.split('_')
        provider = split_name[1][:-2]
        technology = split_name[1][-2:]
        print(provider, " ", technology)

        # === DF for net values ===
        df_net = df.assign(Vendor=provider, Technology=technology, Network='NET')

        filtered_columns_net = df_net[['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA TOTAL_MBYTES_NOCPERF', 'TOTAL_MBYTES_NOCPERF', '% PS_FAIILURE_RRC',
                                       'PS_FAIILURE_RRC', '% PS_FAILURES_RAB', 'PS_FAILURES_RAB', 'LCS_PS_RATE',
                                       'PS_ABNORMAL_RELEASES', 'DELTA TOTAL_ERLANGS_NOCPERF', 'TOTAL_ERLANGS_NOCPERF',
                                       '% CS_FAILURES_RRC', 'CS_FAILURES_RRC', '% CS_FAILURES_RAB', 'CS_FAILURES_RAB',
                                       'LCS_CS_RATE', 'CS_ABNORMAL_RELEASES']]

        df_net = filtered_columns_net.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            #   ,'INTEGRITY': 'Integrity'
            , 'DELTA TOTAL_MBYTES_NOCPERF': 'PS_TRAFF_DELTA'
            , 'TOTAL_MBYTES_NOCPERF': 'PS_TRAFF_GB'
            , '% PS_FAIILURE_RRC': 'PS_RRC_%IA'
            , 'PS_FAIILURE_RRC': 'PS_RRC_FAIL'
            , '% PS_FAILURES_RAB': 'PS_RAB_%IA'
            , 'PS_FAILURES_RAB': 'PS_RAB_FAIL'
            , 'LCS_PS_RATE': 'PS_DROP_%DC'
            , 'PS_ABNORMAL_RELEASES': 'PS_DROP_ABNREL'
            , 'DELTA TOTAL_ERLANGS_NOCPERF': 'CS_TRAFF_DELTA'
            , 'TOTAL_ERLANGS_NOCPERF': 'CS_TRAFF_ERL'
            , '% CS_FAILURES_RRC': 'CS_RRC_%IA'
            , 'CS_FAILURES_RRC': 'CS_RRC_FAIL'
            , '% CS_FAILURES_RAB': 'CS_RAB_%IA'
            , 'CS_FAILURES_RAB': 'CS_RAB_FAIL'
            , 'LCS_CS_RATE': 'CS_DROP_%DC'
            , 'CS_ABNORMAL_RELEASES': 'CS_DROP_ABNREL'
        })

        col_mb_gb = ['PS_TRAFF_GB']
        df_net[col_mb_gb] = df_net[col_mb_gb] / 1024

        # === DF for ATT values ===
        df_ATT = df.assign(Vendor=provider, Technology=technology, Network='ATT')
        filtered_columns_ATT = df_ATT[['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA TRAFFIC_GB_ATT',
                                       'TRAFFIC_GB_ATT',
                                       'DELTA TRAFFIC_AMR_ATT',
                                       'TRAFFIC_AMR_ATT'
                                       ]]

        df_ATT = filtered_columns_ATT.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            #    ,'INTEGRITY': 'Integrity'
            , 'DELTA TRAFFIC_GB_ATT': 'PS_TRAFF_DELTA'
            , 'TRAFFIC_GB_ATT': 'PS_TRAFF_GB'
            , 'DELTA TRAFFIC_AMR_ATT': 'CS_TRAFF_DELTA'
            , 'TRAFFIC_AMR_ATT': 'CS_TRAFF_ERL'
        })
        # === DF for TEF values ===
        df_TEF = df.assign(Vendor=provider, Technology=technology, Network='TEF')
        filtered_columns_TEF = df_TEF[['Network', 'Vendor', 'Technology', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA TRAFFIC_GB_PLMN2',
                                       'TRAFFIC_GB_PLMN2',
                                       'DELTA TRAFFIC_AMR_PLMN2',
                                       'TRAFFIC_AMR_PLMN2'
                                       ]]

        df_TEF = filtered_columns_TEF.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            #    ,'INTEGRITY': 'Integrity'
            , 'DELTA TRAFFIC_GB_PLMN2': 'PS_TRAFF_DELTA'
            , 'TRAFFIC_GB_PLMN2': 'PS_TRAFF_GB'
            , 'DELTA TRAFFIC_AMR_PLMN2': 'CS_TRAFF_DELTA'
            , 'TRAFFIC_AMR_PLMN2': 'CS_TRAFF_ERL'
        })

        dataframes = [df_net, df_ATT, df_TEF]

        df_combinado = pd.concat(dataframes, ignore_index=True, sort=False)
        priority_cols = ['Network', 'Technology', 'Vendor']
        df_combinado = df_combinado[priority_cols + [col for col in df_combinado.columns if col not in priority_cols]]
        #os.makedirs('/usr/local/iquall/mat/shared/sandbox/apps/9jd-guh-r7j/Procesados', exist_ok=True)
        df_combinado.to_csv(f"{current_dir}/Procesados/df_huawei3g.csv",
                            index=False)

def df_huawei4g(**kwargs):
        current_dir = get_work_base_dir(kwargs)
        pattern = os.path.join(current_dir, 'extracted_files', 'ATT_HUAWEI4G_NOC_CLUSTER_*.csv')
        files = glob.glob(pattern)
        if not files:
            print('No se encontraron archivos')
            return
        latest = max(files, key=os.path.getmtime)
        print(f" Archivo encontrado: {latest}")
        df = pd.read_csv(latest)
        file_name = os.path.basename(latest)

        split_name = file_name.split('_')
        provider = split_name[1][:-2]
        technology = split_name[1][-2:]
        print(provider, " ", technology)

        # === DF for net values ===
        df_net = df.assign(Vendor=provider, Technology=technology, Network='NET')

        filtered_columns_net = df_net[
            ['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY', 'DELTA TRAFFIC_DL_UL_MB',
             'TRAFFIC_DL_UL_MB', '% RRC_FAILURES', 'RRC_FAILURES', '% ERAB_FAILURES', 'ERAB_FAILURES', '% S1_FAILURES',
             'S1_FAILURES', '% ERAB_DROPS', 'ERAB_DROPS'
             ]]

        df_net = filtered_columns_net.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            , 'DELTA TRAFFIC_DL_UL_MB': 'PS_TRAFF_DELTA'
            , 'TRAFFIC_DL_UL_MB': 'PS_TRAFF_GB'
            , '% RRC_FAILURES': 'PS_RRC_%IA'
            , 'RRC_FAILURES': 'PS_RRC_FAIL'
            , '% ERAB_FAILURES': 'PS_RAB_%IA'
            , 'ERAB_FAILURES': 'PS_RAB_FAIL'
            , '% S1_FAILURES': 'PS_S1_%IA'
            , 'S1_FAILURES': 'PS_S1_FAIL'
            , '% ERAB_DROPS': 'PS_DROP_%DC'
            , 'ERAB_DROPS': 'PS_DROP_ABNREL'
        })

        col_mb_gb = ['PS_TRAFF_GB']
        df_net[col_mb_gb] = df_net[col_mb_gb] / 1024

        # === DF for ATT values ===
        df_ATT = df.assign(Vendor=provider, Technology=technology, Network='ATT')
        filtered_columns_ATT = df_ATT[
            ['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY', 'DELTA PLMN_1_TRAFFIC_GB'
                , 'PLMN_1_TRAFFIC_GB'
                , '% PLMN_1_ERAB_FAILURES'
                , 'PLMN_1_DCR_1'
                , 'DELTA PLMN_1_VOLTEERLANGS'
                , 'PLMN_1_VOLTEERLANGS'
                , '% PLMN_1_VOLTE_FAILURES'
                , 'PLMN_1_VOLTE_DCR_1'
             ]]

        df_ATT = filtered_columns_ATT.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            , 'DELTA PLMN_1_TRAFFIC_GB': 'PS_TRAFF_DELTA'
            , 'PLMN_1_TRAFFIC_GB': 'PS_TRAFF_GB'
            , '% PLMN_1_ERAB_FAILURES': 'PS_RAB_%IA'
            , 'PLMN_1_DCR_1': 'PS_DROP_%DC'
            , 'DELTA PLMN_1_VOLTEERLANGS': 'CS_TRAFF_DELTA'
            , 'PLMN_1_VOLTEERLANGS': 'CS_TRAFF_ERL'
            , '% PLMN_1_VOLTE_FAILURES': 'CS_RAB_%IA'
            , 'PLMN_1_VOLTE_DCR_1': 'CS_DROP_%DC'
        })
        # === DF for TEF values ===
        df_TEF = df.assign(Vendor=provider, Technology=technology, Network='TEF')
        filtered_columns_TEF = df_TEF[
            ['Network', 'Vendor', 'Technology', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY', 'DELTA PLMN_2_TRAFFIC_GB'
                , 'PLMN_2_TRAFFIC_GB'
                , '% PLMN_2_ERAB_FAILURES'
                , 'PLMN_2_DCR_1'
                , 'DELTA PLMN_2_VOLTEERLANGS'
                , 'PLMN_2_VOLTEERLANGS'
                , '% PLMN_2_VOLTE_FAILURES'
                , 'PLMN_2_VOLTE_DCR_1'
             ]]

        df_TEF = filtered_columns_TEF.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            , 'DELTA PLMN_2_TRAFFIC_GB': 'PS_TRAFF_DELTA'
            , 'PLMN_2_TRAFFIC_GB': 'PS_TRAFF_GB'
            , '% PLMN_2_ERAB_FAILURES': 'PS_RAB_%IA'
            , 'PLMN_2_DCR_1': 'PS_DROP_%DC'
            , 'DELTA PLMN_2_VOLTEERLANGS': 'CS_TRAFF_DELTA'
            , 'PLMN_2_VOLTEERLANGS': 'CS_TRAFF_ERL'
            , '% PLMN_2_VOLTE_FAILURES': 'CS_RAB_%IA'
            , 'PLMN_2_VOLTE_DCR_1': 'CS_DROP_%DC'
        })

        dataframes = [df_net, df_ATT, df_TEF]

        df_combinado = pd.concat(dataframes, ignore_index=True, sort=False)
        priority_cols = ['Network', 'Technology', 'Vendor']
        df_combinado = df_combinado[priority_cols + [col for col in df_combinado.columns if col not in priority_cols]]
        #os.makedirs('/usr/local/iquall/mat/shared/sandbox/apps/9jd-guh-r7j/Procesados', exist_ok=True)

        df_combinado.to_csv(f"{current_dir}/Procesados/df_huawei4g.csv",
                            index=False)

def df_nokia3g(**kwargs):
        current_dir = get_work_base_dir(kwargs)
        pattern = os.path.join(current_dir, 'extracted_files', 'ATT_NOKIA3G_NOC_CLUSTER_*.csv')
        files = glob.glob(pattern)
        if not files:
            print('No se encontraron archivos')
            return
        latest = max(files, key=os.path.getmtime)
        print(f" Archivo encontrado: {latest}")
        df = pd.read_csv(latest)
        file_name = os.path.basename(latest)

        split_name = file_name.split('_')
        provider = split_name[1][:-2]
        technology = split_name[1][-2:]
        print(provider, " ", technology)

        # === DF for net values ===
        df_net = df.assign(Vendor=provider, Technology=technology, Network='NET')

        filtered_columns_net = df_net[
            ['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY', 'DELTA TRAFFIC_TOTAL_GB',
             'TRAFFIC_TOTAL_GB', '% PS_FAILURES_RAB', 'PS_FAILURES_RAB', 'PS_DCR', 'DELTA TRAFFIC_CS', 'TRAFFIC_CS',
             '% CS_FAILURES_RRC', 'CS_FAILURES_RRC', '% CS_FAILURES_RAB', 'CS_FAILURES_RAB', 'DROPS_VOICE']]

        df_net = filtered_columns_net.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            #   ,'INTEGRITY': 'Integrity'
            , 'DELTA TRAFFIC_TOTAL_GB': 'PS_TRAFF_DELTA'
            , 'TRAFFIC_TOTAL_GB': 'PS_TRAFF_GB'
            , '% PS_FAILURES_RAB': 'PS_RAB_%IA'
            , 'PS_FAILURES_RAB': 'PS_RAB_FAIL'
            , 'PS_DCR': 'PS_DROP_%DC'
            , 'DELTA TRAFFIC_CS': 'CS_TRAFF_DELTA'
            , 'TRAFFIC_CS': 'CS_TRAFF_ERL'
            , '% CS_FAILURES_RRC': 'CS_RRC_%IA'
            , 'CS_FAILURES_RRC': 'CS_RRC_FAIL'
            , '% CS_FAILURES_RAB': 'CS_RAB_%IA'
            , 'CS_FAILURES_RAB': 'CS_RAB_FAIL'
            #    ,'VOICE_DCR' : 'CS_DROP_%DC'
            , 'DROPS_VOICE': 'CS_DROP_ABNREL'

        })
        # === DF for ATT values ===
        df_ATT = df.assign(Vendor=provider, Technology=technology, Network='ATT')
        filtered_columns_ATT = df_ATT[['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA TRAFFIC_GB_ATT',
                                       'TRAFFIC_GB_ATT',
                                       'DELTA TRAFFIC_AMR_ATT',
                                       'TRAFFIC_AMR_ATT'
                                       ]]
        df_ATT = filtered_columns_ATT.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            #    ,'INTEGRITY': 'Integrity'
            , 'DELTA TRAFFIC_GB_ATT': 'PS_TRAFF_DELTA'
            , 'TRAFFIC_GB_ATT': 'PS_TRAFF_GB'
            , 'DELTA TRAFFIC_AMR_ATT': 'CS_TRAFF_DELTA'
            , 'TRAFFIC_AMR_ATT': 'CS_TRAFF_ERL'
        })
        # === DF for TEF values ===
        df_TEF = df.assign(Vendor=provider, Technology=technology, Network='TEF')

        filtered_columns_TEF = df_TEF[['Network', 'Vendor', 'Technology', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA TRAFFIC_GB_PLMN2',
                                       'TRAFFIC_GB_PLMN2',
                                       'DELTA TRAFFIC_AMR_PLMN2',
                                       'TRAFFIC_AMR_PLMN2'
                                       ]]
        df_TEF = filtered_columns_TEF.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            #    ,'INTEGRITY': 'Integrity'
            , 'DELTA TRAFFIC_GB_PLMN2': 'PS_TRAFF_DELTA'
            , 'TRAFFIC_GB_PLMN2': 'PS_TRAFF_GB'
            , 'DELTA TRAFFIC_AMR_PLMN2': 'CS_TRAFF_DELTA'
            , 'TRAFFIC_AMR_PLMN2': 'CS_TRAFF_ERL'
        })

        dataframes = [df_net, df_ATT, df_TEF]

        df_combinado = pd.concat(dataframes, ignore_index=True, sort=False)
        priority_cols = ['Network', 'Technology', 'Vendor']
        df_combinado = df_combinado[priority_cols + [col for col in df_combinado.columns if col not in priority_cols]]
        #os.makedirs('/usr/local/iquall/mat/shared/sandbox/apps/9jd-guh-r7j/Procesados', exist_ok=True)

        df_combinado.to_csv(f"{current_dir}/Procesados/df_nokia3g.csv",
                            index=False)



def df_nokia4g(**kwargs):
        current_dir = get_work_base_dir(kwargs)
        pattern = os.path.join(current_dir, 'extracted_files', 'ATT_NOKIA4G_NOC_CLUSTER_*.csv')
        files = glob.glob(pattern)
        if not files:
            print('No se encontraron archivos')
            return
        latest = max(files, key=os.path.getmtime)
        print(f" Archivo encontrado: {latest}")
        df = pd.read_csv(latest)
        file_name = os.path.basename(latest)

        split_name = file_name.split('_')
        provider = split_name[1][:-2]
        technology = split_name[1][-2:]
        print(provider, " ", technology)

        # === DF for net values ===
        df_net = df.assign(Vendor=provider, Technology=technology, Network='NET')

        filtered_columns_net = df_net[['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA TRAFFIC_D_USER_PS_GB', 'TRAFFIC_D_USER_PS_GB', '% RRC_FAILURES',
                                       'RRC_FAILURES', '% ERAB_FAILURES', 'ERAB_FAILURES', '% S1_FAILURES',
                                       'S1_FAILURES', '% ERAB_DROPS', 'ERAB_DROPS']]
        df_net = filtered_columns_net.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            #   ,'INTEGRITY': 'Integrity'
            , 'DELTA TRAFFIC_D_USER_PS_GB': 'PS_TRAFF_DELTA'
            , 'TRAFFIC_D_USER_PS_GB': 'PS_TRAFF_GB'
            , '% RRC_FAILURES': 'PS_RRC_%IA'
            , 'RRC_FAILURES': 'PS_RRC_FAIL'
            , '% ERAB_FAILURES': 'PS_RAB_%IA'
            , 'ERAB_FAILURES': 'PS_RAB_FAIL'
            , '% S1_FAILURES': 'PS_S1_%IA'
            , 'S1_FAILURES': 'PS_S1_FAIL'
            , '% ERAB_DROPS': 'PS_DROP_%DC'
            , 'ERAB_DROPS': 'PS_DROP_ABNREL'
        })
        # === DF for ATT values ===
        df_ATT = df.assign(Vendor=provider, Technology=technology, Network='ATT',
                           SUM_PLMN_TRAFFIC_GB=df['PLMN_0_TRAFFIC_GB'] + df['PLMN_1_TRAFFIC_GB'])
        filtered_columns_ATT = df_ATT[
            ['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY', 'DELTA PLMN_0_TRAFFIC_GB',
             'SUM_PLMN_TRAFFIC_GB']]
        df_ATT = filtered_columns_ATT.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            , 'DELTA PLMN_0_TRAFFIC_GB': 'PS_TRAFF_DELTA'
            , 'SUM_PLMN_TRAFFIC_GB': 'PS_TRAFF_GB'
        })
        # === DF for TEF values ===
        df_TEF = df.assign(Vendor=provider, Technology=technology, Network='TEF')

        filtered_columns_TEF = df_TEF[['Network', 'Vendor', 'Technology', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA PLMN_2_TRAFFIC_GB', 'PLMN_2_TRAFFIC_GB'
                                       ]]
        df_TEF = filtered_columns_TEF.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            , 'DELTA PLMN_2_TRAFFIC_GB': 'PS_TRAFF_DELTA'
            , 'PLMN_2_TRAFFIC_GB': 'PS_TRAFF_GB'
        })

        dataframes = [df_net, df_ATT, df_TEF]

        df_combinado = pd.concat(dataframes, ignore_index=True, sort=False)
        priority_cols = ['Network', 'Technology', 'Vendor']
        df_combinado = df_combinado[priority_cols + [col for col in df_combinado.columns if col not in priority_cols]]
        #os.makedirs('/usr/local/iquall/mat/shared/sandbox/apps/9jd-guh-r7j/Procesados', exist_ok=True)

        df_combinado.to_csv(f"{current_dir}/Procesados/df_nokia4g.csv",
                            index=False)



def df_samsung4g(**kwargs):
        current_dir = get_work_base_dir(kwargs)
        pattern = os.path.join(current_dir, 'extracted_files', 'ATT_SAMSUNG4G_NOC_CLUSTER_*.csv')
        files = glob.glob(pattern)
        if not files:
            print('No se encontraron archivos')
            return
        latest = max(files, key=os.path.getmtime)
        print(f" Archivo encontrado: {latest}")
        df = pd.read_csv(latest)
        file_name = os.path.basename(latest)

        split_name = file_name.split('_')
        provider = split_name[1][:-2]
        technology = split_name[1][-2:]
        print(provider, " ", technology)

        # === DF for net values ===
        df_net = df.assign(Vendor=provider, Technology=technology, Network='NET')

        filtered_columns_net = df_net[['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA TRAFFIC_D_USER_PS_GB', 'TRAFFIC_D_USER_PS_GB', '% RRC_FAILURES',
                                       'RRC_FAILURES', '% ERAB_FAILURES', 'ERAB_FAILURES', '% S1_FAILURES',
                                       'S1_FAILURES', '% RETAINABILITY_NUM', 'RETAINABILITY_NUM']]
        df_net = filtered_columns_net.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            #   ,'INTEGRITY': 'Integrity'
            , 'DELTA TRAFFIC_D_USER_PS_GB': 'PS_TRAFF_DELTA'
            , 'TRAFFIC_D_USER_PS_GB': 'PS_TRAFF_GB'
            , '% RRC_FAILURES': 'PS_RRC_%IA'
            , 'RRC_FAILURES': 'PS_RRC_FAIL'
            , '% ERAB_FAILURES': 'PS_RAB_%IA'
            , 'ERAB_FAILURES': 'PS_RAB_FAIL'
            , '% S1_FAILURES': 'PS_S1_%IA'
            , 'S1_FAILURES': 'PS_S1_FAIL'
            , '% RETAINABILITY_NUM': 'PS_DROP_%DC'
            , 'RETAINABILITY_NUM': 'PS_DROP_ABNREL'
        })
        # === DF for ATT values ===
        df_ATT = df.assign(Vendor=provider, Technology=technology, Network='ATT',
                           SUM_PLMN_TRAFFIC_GB=df['PLMN_0_TRAFFIC_GB'] + df['PLMN_1_TRAFFIC_GB'])
        filtered_columns_ATT = df_ATT[['Network', 'Technology', 'Vendor', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA PLMN_0_TRAFFIC_GB', 'SUM_PLMN_TRAFFIC_GB'
                                       ]]
        df_ATT = filtered_columns_ATT.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            , 'DELTA PLMN_0_TRAFFIC_GB': 'PS_TRAFF_DELTA'
            , 'SUM_PLMN_TRAFFIC_GB': 'PS_TRAFF_GB'
        })
        # === DF for TEF values ===
        df_TEF = df.assign(Vendor=provider, Technology=technology, Network='TEF')

        filtered_columns_TEF = df_TEF[['Network', 'Vendor', 'Technology', 'NOC_CLUSTER', 'DATE', 'TIME', 'INTEGRITY',
                                       'DELTA PLMN_2_TRAFFIC_GB', 'PLMN_2_TRAFFIC_GB'
                                       ]]
        df_TEF = filtered_columns_TEF.rename(columns={
            'NOC_CLUSTER': 'Noc_Cluster'
            , 'DATE': 'Date'
            , 'TIME': 'Time'
            , 'DELTA PLMN_2_TRAFFIC_GB': 'PS_TRAFF_DELTA'
            , 'PLMN_2_TRAFFIC_GB': 'PS_TRAFF_GB'
        })

        dataframes = [df_net, df_ATT, df_TEF]

        df_combinado = pd.concat(dataframes, ignore_index=True, sort=False)
        priority_cols = ['Network', 'Technology', 'Vendor']
        df_combinado = df_combinado[priority_cols + [col for col in df_combinado.columns if col not in priority_cols]]
        #os.makedirs('/usr/local/iquall/mat/shared/sandbox/apps/9jd-guh-r7j/Procesados', exist_ok=True)

        df_combinado.to_csv(f"{current_dir}/Procesados/df_samsung4g.csv",
                            index=False)



def df_merged(**kwargs):
    # Ruta donde están los archivos CSV

    current_dir = get_work_base_dir(kwargs)
    folder_path = os.path.join(current_dir, 'Procesados')

    #folder_path = '/usr/local/iquall/mat/shared/sandbox/apps/9jd-guh-r7j/Procesados'
    # folder_path = Path(r"../Archivos_por_hora_desc")
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
        os.makedirs(f'{current_dir}/DF_Consolidado', exist_ok=True)
        #os.makedirs('/usr/local/iquall/mat/shared/sandbox/apps/9jd-guh-r7j/DF_Consolidado', exist_ok=True)
        combined_df = combined_df[nuevo_orden]
        combined_df = combined_df.round(2)
        combined_df.to_csv(f'{current_dir}/DF_Consolidado/df_resultante.csv',
                           index=False)
        homologado = f'{current_dir}/DF_Consolidado/df_resultante.csv'

def clean_value(value):
    """
    Convierte valores NaN/NaT a None para que GraphQL los acepte como null.
    """
    if pd.isna(value):
        return None

    # Convierte tipos numpy/pandas a tipos nativos de Python
    if hasattr(value, "item"):
        return value.item()

    return value


SERVER_SORT_FIELDS = {
    "severity_score": "Severity_Score",
    "crit_count": "Crit_Count",
    "complete_flag": "Complete_Flag",
    "integrity_health_pct": "Integrity_Health_Pct",
}

SEVERITY_METRICS = [
    "PS_RRC__IA",
    "PS_RAB__IA",
    "PS_S1__IA",
    "PS_DROP__DC",
    "CS_RRC__IA",
    "CS_RAB__IA",
    "CS_DROP__DC",
]

DEFAULT_THRESHOLDS = {
    "PS_RRC__IA": {
        "orientation": "lower_is_better",
        "thresholds": {"excelente": 1.5, "bueno": 2.0, "regular": 3.0, "critico": 5.0},
    },
    "PS_RAB__IA": {
        "orientation": "lower_is_better",
        "thresholds": {"excelente": 1.5, "bueno": 2.0, "regular": 3.0, "critico": 5.0},
    },
    "PS_S1__IA": {
        "orientation": "lower_is_better",
        "thresholds": {"excelente": 1.5, "bueno": 2.0, "regular": 3.0, "critico": 5.0},
    },
    "PS_DROP__DC": {
        "orientation": "lower_is_better",
        "thresholds": {"excelente": 1.5, "bueno": 2.0, "regular": 3.0, "critico": 5.0},
    },
    "CS_RRC__IA": {
        "orientation": "lower_is_better",
        "thresholds": {"excelente": 1.5, "bueno": 2.0, "regular": 3.0, "critico": 5.0},
    },
    "CS_RAB__IA": {
        "orientation": "lower_is_better",
        "thresholds": {"excelente": 1.5, "bueno": 2.0, "regular": 3.0, "critico": 5.0},
    },
    "CS_DROP__DC": {
        "orientation": "lower_is_better",
        "thresholds": {"excelente": 1.5, "bueno": 2.0, "regular": 3.0, "critico": 5.0},
    },
}

THRESHOLD_CONFIG_TABLE = "Resources_DashboardThresholdConfig"

THRESHOLD_KEY_BY_DB_FIELD = {
    "PS_RRC__IA": "ps_rrc_ia_percent",
    "PS_RAB__IA": "ps_rab_ia_percent",
    "PS_S1__IA": "ps_s1_ia_percent",
    "PS_DROP__DC": "ps_drop_dc_percent",
    "CS_RRC__IA": "cs_rrc_ia_percent",
    "CS_RAB__IA": "cs_rab_ia_percent",
    "CS_DROP__DC": "cs_drop_dc_percent",
}


def to_float(value):
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def metric_severity_level(metric, raw_value, network=None, thresholds_snapshot=None):
    value = to_float(raw_value)
    if value is None:
        return 0

    cfg = metric_threshold_config(metric, network, thresholds_snapshot)
    thresholds = cfg.get("thresholds") or {}
    orientation = cfg.get("orientation", "lower_is_better")

    exc = to_float(thresholds.get("excelente"))
    bue = to_float(thresholds.get("bueno"))
    reg = to_float(thresholds.get("regular"))
    cri = to_float(thresholds.get("critico") or cfg.get("critical"))
    if cri is None:
        return 0

    if orientation == "higher_is_better":
        if value <= cri:
            return 4
        if reg is not None and value <= reg:
            return 3
        if bue is not None and value <= bue:
            return 2
        if exc is not None and value <= exc:
            return 1
        return 0

    if value >= cri:
        return 4
    if reg is not None and value >= reg:
        return 3
    if bue is not None and value >= bue:
        return 2
    if exc is not None and value >= exc:
        return 1
    return 0


def row_severity_score(row, thresholds_snapshot=None):
    score = 0
    network = None if pd.isna(row.get("Network")) else str(row.get("Network"))
    for metric in SEVERITY_METRICS:
        score += metric_severity_level(metric, row.get(metric), network, thresholds_snapshot)
    return score


def row_crit_count(row, thresholds_snapshot=None):
    count = 0
    network = None if pd.isna(row.get("Network")) else str(row.get("Network"))
    for metric in SEVERITY_METRICS:
        if metric_severity_level(metric, row.get(metric), network, thresholds_snapshot) >= 4:
            count += 1
    return count


def metric_threshold_config(metric, network=None, thresholds_snapshot=None):
    friendly_key = THRESHOLD_KEY_BY_DB_FIELD.get(metric, metric)
    snapshot = thresholds_snapshot or {}
    profiles = snapshot.get("profiles") or {}
    main_profile = profiles.get("main") or {}
    severity = main_profile.get("severity") or snapshot.get("severity") or {}
    cfg = severity.get(friendly_key)

    if cfg:
        default_cfg = cfg.get("default") or cfg
        per_network = cfg.get("per_network") or {}
        if network and network in per_network:
            net_cfg = per_network.get(network) or {}
            merged = dict(default_cfg)
            merged.update(net_cfg)
            if "thresholds" not in merged:
                merged["thresholds"] = default_cfg.get("thresholds") or {}
            if "orientation" not in merged:
                merged["orientation"] = default_cfg.get("orientation", "lower_is_better")
            return merged
        return default_cfg

    return DEFAULT_THRESHOLDS.get(metric) or {}


def compute_previous_week_window(fecha):
    selected_dt = datetime.strptime(str(fecha), "%Y-%m-%d")
    current_monday = selected_dt - timedelta(days=selected_dt.weekday())
    prev_monday = current_monday - timedelta(days=7)
    prev_sunday = current_monday - timedelta(days=1)
    return prev_monday.strftime("%Y-%m-%d"), prev_sunday.strftime("%Y-%m-%d")


def fetch_integrity_baseline_map_for_insert(mat, fecha, key_rows, job):
    if not key_rows:
        return {}

    try:
        prev_monday, prev_sunday = compute_previous_week_window(fecha)
    except Exception as exc:
        job.log.warning(f"No se pudo calcular ventana baseline para fecha={fecha}: {exc}")
        return {}

    query = """
    query getIntegrityBaseline(
      $where: Resources_dashboardMaster_bool_exp!,
      $limit: Int!,
      $offset: Int!
    ) {
      Resources_dashboardMaster(
        where: $where,
        limit: $limit,
        offset: $offset,
        order_by: [{Date: asc}, {Time: asc}]
      ) {
        Network
        Vendor
        Noc_Cluster
        Technology
        INTEGRITY
      }
    }
    """

    baseline_sums = {}
    baseline_counts = {}
    chunk_size = 100
    page_size = 1000

    for chunk_start in range(0, len(key_rows), chunk_size):
        chunk = key_rows[chunk_start:chunk_start + chunk_size]
        key_or = [
            {
                "_and": [
                    {"Network": {"_eq": row["Network"]}},
                    {"Vendor": {"_eq": row["Vendor"]}},
                    {"Noc_Cluster": {"_eq": row["Noc_Cluster"]}},
                    {"Technology": {"_eq": row["Technology"]}},
                ]
            }
            for row in chunk
        ]

        where = {
            "_and": [
                {"Date": {"_gte": prev_monday, "_lte": prev_sunday}},
                {"_or": key_or},
            ]
        }

        offset = 0
        while True:
            result = mat.graphQL.execute(
                operation=query,
                variables={"where": where, "limit": page_size, "offset": offset}
            )

            if result.get("errors"):
                raise Exception(
                    f"Error consultando baseline de integridad en GraphQL: {result.get('errors')}"
                )

            rows = (
                result
                .get("data", {})
                .get("Resources_dashboardMaster", [])
            )

            if not rows:
                break

            for item in rows:
                integrity = to_float(item.get("INTEGRITY"))
                if integrity is None:
                    continue
                key = (
                    item.get("Network"),
                    item.get("Vendor"),
                    item.get("Noc_Cluster"),
                    item.get("Technology"),
                )
                baseline_sums[key] = baseline_sums.get(key, 0.0) + integrity
                baseline_counts[key] = baseline_counts.get(key, 0) + 1

            if len(rows) < page_size:
                break
            offset += page_size

    baseline_map = {}
    for key, total in baseline_sums.items():
        count = baseline_counts.get(key) or 1
        baseline_map[key] = total / count

    job.log.info(
        f"Baseline integridad {prev_monday}..{prev_sunday} calculado para {len(baseline_map)} combinaciones."
    )
    return baseline_map


def fetch_active_threshold_config_for_insert(mat, job, profile="main"):
    query = """
    query getThresholdConfig($where: Resources_DashboardThresholdConfig_bool_exp!, $limit: Int!) {
      Resources_DashboardThresholdConfig(
        where: $where,
        limit: $limit,
        order_by: [{updated_at: desc}, {metadata_created: desc}]
      ) {
        profile
        config_json
        config_hash
        active
        updated_at
      }
    }
    """
    where = {
        "_and": [
            {"profile": {"_eq": profile}},
            {"active": {"_in": ["true", "1", "TRUE", "True", "yes", "YES"]}},
        ]
    }
    result = mat.graphQL.execute(operation=query, variables={"where": where, "limit": 1})
    if result.get("errors"):
        job.log.warning(f"No se pudo consultar config de umbrales activa: {result.get('errors')}")
        return None

    rows = (
        result
        .get("data", {})
        .get(THRESHOLD_CONFIG_TABLE, [])
    )
    if not rows:
        job.log.warning("No existe config activa de umbrales en GraphQL; se usara fallback local.")
        return None

    row = rows[0]
    try:
        config = json.loads(row.get("config_json") or "{}")
    except Exception as exc:
        job.log.warning(f"No se pudo parsear config_json de umbrales; se usara fallback local: {exc}")
        return None

    job.log.info(
        "Config de umbrales activa cargada: "
        f"profile={row.get('profile')} hash={row.get('config_hash')} updated_at={row.get('updated_at')}"
    )
    return config


def add_dashboard_master_computed_columns(df_insert, mat, job):
    required_cols = ["Date", "Network", "Vendor", "Noc_Cluster", "Technology"]
    missing_cols = [col for col in required_cols if col not in df_insert.columns]
    if missing_cols:
        job.log.warning(f"No se calculan columnas optimizadas. Faltan columnas: {missing_cols}")
        return df_insert

    out = df_insert.copy()
    thresholds_snapshot = fetch_active_threshold_config_for_insert(mat, job, profile="main")
    key_cols = ["Network", "Vendor", "Noc_Cluster", "Technology"]
    baseline_maps_by_date = {}

    valid_key_df = out.dropna(subset=required_cols)
    for fecha in sorted(valid_key_df["Date"].astype(str).unique()):
        date_df = valid_key_df[valid_key_df["Date"].astype(str) == fecha]
        key_rows = (
            date_df[key_cols]
            .astype(str)
            .drop_duplicates()
            .to_dict(orient="records")
        )
        baseline_maps_by_date[fecha] = fetch_integrity_baseline_map_for_insert(mat, fecha, key_rows, job)

    severity_scores = []
    crit_counts = []
    complete_flags = []
    integrity_health_pcts = []

    for row in out.to_dict(orient="records"):
        severity_score = row_severity_score(row, thresholds_snapshot)
        crit_count = row_crit_count(row, thresholds_snapshot)
        integrity = to_float(row.get("INTEGRITY"))
        complete_flag = 0 if integrity is not None and integrity >= 80 else 1

        baseline_key = (
            None if pd.isna(row.get("Network")) else str(row.get("Network")),
            None if pd.isna(row.get("Vendor")) else str(row.get("Vendor")),
            None if pd.isna(row.get("Noc_Cluster")) else str(row.get("Noc_Cluster")),
            None if pd.isna(row.get("Technology")) else str(row.get("Technology")),
        )
        fecha = None if pd.isna(row.get("Date")) else str(row.get("Date"))
        baseline = to_float((baseline_maps_by_date.get(fecha) or {}).get(baseline_key))

        integrity_health_pct = None
        if integrity is not None and baseline is not None and baseline > 0:
            integrity_health_pct = max(0.0, min(100.0, (integrity / baseline) * 100.0))

        severity_scores.append(int(severity_score))
        crit_counts.append(int(crit_count))
        complete_flags.append(int(complete_flag))
        integrity_health_pcts.append(integrity_health_pct)

    out[SERVER_SORT_FIELDS["severity_score"]] = severity_scores
    out[SERVER_SORT_FIELDS["crit_count"]] = crit_counts
    out[SERVER_SORT_FIELDS["complete_flag"]] = complete_flags
    out[SERVER_SORT_FIELDS["integrity_health_pct"]] = integrity_health_pcts

    job.log.info(
        "Columnas optimizadas calculadas: "
        f"{list(SERVER_SORT_FIELDS.values())}"
    )
    return out


def insert_dashboard_master_graphql(**kwargs):
    job = kwargs.get("job")
    mat = MATClient()

    if not is_dashboard_master_available(mat, job):
        job.log.warning(
            "Recurso dashboardMaster no disponible en esta ejecucion. "
            "Se omite carga GraphQL y continua el flujo para enviar correo."
        )
        return

    current_dir = get_work_base_dir(kwargs)
    file_path = os.path.join(current_dir, "DF_Consolidado", "df_resultante.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No existe el archivo consolidado: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        job.log.info("El archivo consolidado está vacío. No hay registros para insertar.")
        return

    pk_cols = [
        "Network",
        "Technology",
        "Vendor",
        "Noc_Cluster",
        "Date",
        "Time"
    ]

    required_cols = [
        "Network",
        "Technology",
        "Vendor",
        "Noc_Cluster",
        "Date",
        "Time"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise Exception(f"Faltan columnas requeridas en el CSV consolidado: {missing_cols}")

    before = len(df)
    df = df.drop_duplicates(subset=pk_cols, keep="last")
    after = len(df)

    job.log.info(f"Registros originales: {before}")
    job.log.info(f"Registros después de eliminar duplicados internos: {after}")
    
    def build_pk_from_values(values):
        return "||".join("" if pd.isna(value) else str(value).strip() for value in values)

    df["__pk"] = df.apply(
        lambda row: build_pk_from_values([row[col] for col in pk_cols]),
        axis=1
    )

    query_existing = """
    query getExistingDashboardMaster($where: Resources_dashboardMaster_bool_exp!) {
      Resources_dashboardMaster(where: $where) {
        Network
        Technology
        Vendor
        Noc_Cluster
        Date
        Time
      }
    }
    """
    
    def chunks(items, size):
        for i in range(0, len(items), size):
            yield items[i:i + size]
    
    pk_rows = (
        df[pk_cols]
        .dropna(subset=pk_cols)
        .drop_duplicates()
        .to_dict(orient="records")
    )
    
    existing_keys = set()
    chunk_size = 100
    
    for chunk in chunks(pk_rows, chunk_size):
        where = {
            "_or": [
                {
                    "Network": {"_eq": str(row["Network"]).strip()},
                    "Technology": {"_eq": str(row["Technology"]).strip()},
                    "Vendor": {"_eq": str(row["Vendor"]).strip()},
                    "Noc_Cluster": {"_eq": str(row["Noc_Cluster"]).strip()},
                    "Date": {"_eq": str(row["Date"]).strip()},
                    "Time": {"_eq": str(row["Time"]).strip()},
                }
                for row in chunk
            ]
        }
    
        result_existing = mat.graphQL.execute(
            operation=query_existing,
            variables={"where": where}
        )
    
        if result_existing.get("errors"):
            raise Exception(
                f"Error consultando registros existentes en GraphQL: {result_existing.get('errors')}"
            )
    
        existing_rows = (
            result_existing
            .get("data", {})
            .get("Resources_dashboardMaster", [])
        )
    
        for existing_row in existing_rows:
            existing_keys.add(
                build_pk_from_values([existing_row.get(col) for col in pk_cols])
            )

    before_existing_filter = len(df)

    df = df[~df["__pk"].isin(existing_keys)].copy()

    after_existing_filter = len(df)

    job.log.info(f"Registros antes de filtrar existentes en GraphQL: {before_existing_filter}")
    job.log.info(f"Registros nuevos para insertar: {after_existing_filter}")
    job.log.info(f"Registros omitidos porque ya existian: {before_existing_filter - after_existing_filter}")

    df = df.drop(columns=["__pk"])

    if df.empty:
        job.log.info("Todos los registros del CSV consolidado ya existen en GraphQL. No hay registros nuevos para insertar.")
        return

    graphql_column_map = {
        "Network": "Network",
        "Technology": "Technology",
        "Vendor": "Vendor",
        "Noc_Cluster": "Noc_Cluster",
        "Date": "Date",
        "Time": "Time",
        "INTEGRITY": "INTEGRITY",

        "PS_TRAFF_DELTA": "PS_TRAFF_DELTA",
        "PS_TRAFF_GB": "PS_TRAFF_GB",

        "PS_RRC_%IA": "PS_RRC__IA",
        "PS_RRC_FAIL": "PS_RRC_FAIL",

        "PS_RAB_%IA": "PS_RAB__IA",
        "PS_RAB_FAIL": "PS_RAB_FAIL",

        "PS_S1_%IA": "PS_S1__IA",
        "PS_S1_FAIL": "PS_S1_FAIL",

        "PS_DROP_%DC": "PS_DROP__DC",
        "PS_DROP_ABNREL": "PS_DROP_ABNREL",

        "CS_TRAFF_DELTA": "CS_TRAFF_DELTA",
        "CS_TRAFF_ERL": "CS_TRAFF_ERL",

        "CS_RRC_%IA": "CS_RRC__IA",
        "CS_RRC_FAIL": "CS_RRC_FAIL",

        "CS_RAB_%IA": "CS_RAB__IA",
        "CS_RAB_FAIL": "CS_RAB_FAIL",

        "CS_DROP_%DC": "CS_DROP__DC",
        "CS_DROP_ABNREL": "CS_DROP_ABNREL",
        "Archivo_Fuente": "Archivo_Fuente",
        "Fecha_Ejecucion": "Fecha_Ejecucion"
    }

    # Tomar solo columnas que realmente existen en el CSV
    available_cols = [
        csv_col for csv_col in graphql_column_map.keys()
        if csv_col in df.columns
    ]

    missing_from_csv = [
        csv_col for csv_col in graphql_column_map.keys()
        if csv_col not in df.columns
    ]

    job.log.info(f"Columnas disponibles en CSV para insertar: {available_cols}")
    job.log.info(f"Columnas del mapeo que NO existen en el CSV: {missing_from_csv}")

    df_insert = df[available_cols].rename(columns=graphql_column_map)

    job.log.info(f"Columnas después del rename: {list(df_insert.columns)}")

    # Validar que ya no existan columnas con %
    percent_cols = [col for col in df_insert.columns if "%" in col]

    if percent_cols:
        raise Exception(
            f"Después del rename todavía existen columnas con %. "
            f"Revisa el mapeo: {percent_cols}"
        )

    # Consultar campos válidos del input GraphQL
    df_insert = add_dashboard_master_computed_columns(df_insert, mat, job)

    schema_input_query = """
    query getInputType($typeName: String!) {
      __type(name: $typeName) {
        name
        inputFields {
          name
        }
      }
    }
    """

    schema_input_result = mat.graphQL.execute(
        operation=schema_input_query,
        variables={"typeName": "Resources_dashboardMaster_insert_input"}
    )

    job.log.info("Schema insert_input:")
    job.log.info(json.dumps(schema_input_result, indent=2))

    if schema_input_result.get("errors"):
        raise Exception(f"Error consultando schema GraphQL: {schema_input_result.get('errors')}")

    input_fields = (
        schema_input_result
        .get("data", {})
        .get("__type", {})
        .get("inputFields", [])
    )

    graphql_fields = [field["name"] for field in input_fields]

    job.log.info(f"Campos válidos en GraphQL insert_input: {graphql_fields}")

    # Dejar solo columnas que existan en GraphQL
    cols_before_filter = list(df_insert.columns)

    df_insert = df_insert[
        [col for col in df_insert.columns if col in graphql_fields]
    ]

    removed_by_graphql_filter = [
        col for col in cols_before_filter
        if col not in df_insert.columns
    ]

    job.log.info(f"Columnas removidas porque no existen en GraphQL: {removed_by_graphql_filter}")
    job.log.info(f"Columnas finales enviadas a GraphQL: {list(df_insert.columns)}")

    if df_insert.empty:
        raise Exception("Después de filtrar columnas válidas de GraphQL, no quedó información para insertar.")

    # Convertir NaN a None
    records = []

    for row in df_insert.to_dict(orient="records"):
        clean_row = {
            key: clean_value(value)
            for key, value in row.items()
        }
        records.append(clean_row)

    if not records:
        job.log.info("No hay registros válidos para insertar.")
        return

    job.log.info("Primer registro que se enviará a GraphQL:")
    job.log.info(json.dumps(records[0], indent=2))

    mutation_insert = """
    mutation insertDashboardMaster($objects: [Resources_dashboardMaster_insert_input!]!) {
      insert_Resources_dashboardMaster(objects: $objects) {
        affected_rows
      }
    }
    """

    batch_size = 500
    total_inserted = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]

        job.log.info(f"Insertando batch {i} - {i + len(batch)}")

        result = mat.graphQL.execute(
            operation=mutation_insert,
            variables={
                "objects": batch
            }
        )

        job.log.info(f"Resultado batch {i} - {i + len(batch)}:")
        job.log.info(json.dumps(result, indent=2))

        if result.get("errors"):
            raise Exception(
                f"Error insertando batch {i} - {i + len(batch)}: {result.get('errors')}"
            )

        affected_rows = (
            result
            .get("data", {})
            .get("insert_Resources_dashboardMaster", {})
            .get("affected_rows", 0)
        )

        total_inserted += affected_rows

    job.log.info(f"Total de registros insertados en Resources_dashboardMaster: {total_inserted}")
    
def send_notification(**kwargs):
    import os
    import re
    import glob
    import shutil
    from datetime import datetime, timedelta

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

    # Leer destinatarios desde UI
    params = getForm(kwargs) or {}
    data = params.get("data") or {}
    ui_email_value = data.get("emailToSend", "")

    dest_ui = _parse_emails(ui_email_value)

    # Fallback si no viene nada desde la UI
    hard_dest = [
        "salvador.caracoza@innovasolutions.com",
    ]

    dest = dest_ui if dest_ui else hard_dest

    folder_path = f"{current_dir}/Procesados"
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
    files = f"{current_dir}/DF_Consolidado/df_resultante.csv"

    ATTSendEmail(
        dest=dest,
        subj=subj,
        msg=msg,
        files=files
    )

    shutil.rmtree(current_dir, ignore_errors=True)
    
########################################################################################################################
###############################################                     ####################################################
###############################################  TASKS DEFINITIONS  ####################################################
###############################################                     ####################################################
########################################################################################################################


init = MATInstanceInitOperator(task_id='MAT_Inicializacion', dag=dag)
download_sftp_files = MATPythonOperator(task_id="descarga_archivos_sftp",python_callable=download_sftp_files,retries=6,retry_delay=timedelta(minutes=5),dag=dag)
extract_sftp_files = MATPythonOperator(task_id="extraccion_archivos__sftp", python_callable = extraction_batch, dag=dag)
df_ericsson3g = MATPythonOperator(task_id="df_ericsson3g", python_callable = df_ericsson3g, dag=dag)
df_ericsson4g = MATPythonOperator(task_id="df_ericsson4g", python_callable = df_ericsson4g, dag=dag)
df_huawei3g = MATPythonOperator(task_id="df_huawei3g", python_callable = df_huawei3g, dag=dag)
df_huawei4g = MATPythonOperator(task_id="df_huawei4g", python_callable = df_huawei4g, dag=dag)
df_nokia3g = MATPythonOperator(task_id="df_nokia3g", python_callable = df_nokia3g, dag=dag)
df_nokia4g = MATPythonOperator(task_id="df_nokia4g", python_callable = df_nokia4g, dag=dag)
df_samsung4g = MATPythonOperator(task_id="df_samsung4g", python_callable = df_samsung4g, dag=dag)
df_consolidado = MATPythonOperator(task_id="consolidacion_final", python_callable = df_merged, dag=dag)
df_send_mail = MATPythonOperator(task_id="enviar_email", python_callable = send_notification, dag=dag)
df_insert_graphql = MATPythonOperator(
    task_id="insertar_dashboard_master_graphql",
    python_callable=insert_dashboard_master_graphql,
    dag=dag
)

end = MATInstanceExitOperator(task_id= 'MAT_Finalizar', dag=dag)


########################################################################################################################
################################################                  ######################################################
################################################  TASKS WORKFLOW  ######################################################
################################################                  ######################################################
########################################################################################################################

init >> download_sftp_files >> extract_sftp_files >> df_ericsson3g >> df_consolidado >> df_insert_graphql >> df_send_mail >> end
extract_sftp_files >> df_ericsson4g >> df_consolidado
extract_sftp_files >> df_huawei3g >> df_consolidado
extract_sftp_files >> df_huawei4g >> df_consolidado
extract_sftp_files >> df_nokia3g >> df_consolidado
extract_sftp_files >> df_nokia4g >> df_consolidado
extract_sftp_files >> df_samsung4g >> df_consolidado


