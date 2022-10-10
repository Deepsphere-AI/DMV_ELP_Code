import pandas as pd
import numpy as np
import tensorflow as tf
from DMV_Text_Classification import ClassificationModels
from DMV_Pattern_Denial import Pattern_Denial
# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel
from random import randint
import time
import json
import gcsfs
import h5py
from google.cloud import storage,bigquery,secretmanager
from io import StringIO 
import boto3
import os
import datetime
import traceback

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def ELP_Validation(request):
    """Responds to any HTTP request.
    """
    try:
        vAR_output = pd.DataFrame()
        vAR_request_url = 's3://s3-us-west-2-elp/batch/simpligov/new/ELP_Project_Request/ELP_Batch_Input_Small.csv'
        vAR_batch_elp_configuration = pd.read_csv(vAR_request_url)
        vAR_number_of_configuration = len(vAR_batch_elp_configuration)
        vAR_file_path = Upload_Request_GCS(vAR_batch_elp_configuration)
        for elp_idx in  range(vAR_number_of_configuration):
            start_time = time.time()
            vAR_result_message = ""
            vAR_input_text = vAR_batch_elp_configuration['CONFIGURATION'][elp_idx].replace('/','')
            vAR_vin = vAR_batch_elp_configuration['VIN'][elp_idx]

            if len(vAR_input_text)>7:
                return {'Error Message':'ELP Configuration can not be more than 7 characters'}

            vAR_profanity_result,vAR_result_message = Profanity_Words_Check(vAR_input_text)
            if not vAR_profanity_result:
                vAR_message_level_1 = "Level 1 Accepted"
            elif vAR_profanity_result:
                vAR_message_level_1 = vAR_result_message

            vAR_regex_result,vAR_pattern = Pattern_Denial(vAR_input_text)
            if not vAR_regex_result:
                vAR_message_level_2 = "Denied - Similar to " +vAR_pattern+ " Pattern"
            elif vAR_regex_result:
                vAR_message_level_2 = "Level 2 Accepted"

                
            vAR_result,vAR_result_data,vAR_result_target_sum = LSTM_Model_Result(vAR_input_text)
            vAR_result_data = vAR_result_data.to_json(orient='records')
            vAR_random_id = Random_Id_Generator()
            if vAR_result_target_sum>20:
                vAR_message_level_3 = "Configuration Denied, Since the profanity probability exceeds the threshold"
            else:
                vAR_message_level_3 = "Level 3 Accepted"
            vAR_response_time = round(time.time() - start_time,2)

            vAR_final_result =  {"1st Level(Direct Profanity)":{"Is accepted":not vAR_profanity_result,"Message":vAR_message_level_1},
            "2nd Level(Denied Pattern)":{"Is accepted":vAR_regex_result,"Message":vAR_message_level_2},
            "3rd Level(Model Prediction)":{"Is accepted":vAR_result,"Message":vAR_message_level_3,"Profanity Classification":json.loads(vAR_result_data),
            'Sum of all Categories':vAR_result_target_sum},
            'Order Id':vAR_random_id,'Configuration':vAR_input_text,
            'Response time':str(vAR_response_time)+" secs","Vehicle Id Number(VIN)":vAR_vin}

            print('vAR Result - ',vAR_final_result)
            
            vAR_final_result['3rd Level(Model Prediction)']['Profanity Classification'][0]['Toxic'] = str(vAR_final_result['3rd Level(Model Prediction)']['Profanity Classification'][0]['Toxic'])
            vAR_final_result['3rd Level(Model Prediction)']['Profanity Classification'][0]['Severe Toxic'] = str(vAR_final_result['3rd Level(Model Prediction)']['Profanity Classification'][0]['Severe Toxic'])
            vAR_final_result['3rd Level(Model Prediction)']['Profanity Classification'][0]['Obscene'] = str(vAR_final_result['3rd Level(Model Prediction)']['Profanity Classification'][0]['Obscene'])
            vAR_final_result['3rd Level(Model Prediction)']['Profanity Classification'][0]['Threat'] = str(vAR_final_result['3rd Level(Model Prediction)']['Profanity Classification'][0]['Threat'])
            vAR_final_result['3rd Level(Model Prediction)']['Profanity Classification'][0]['Insult'] = str(vAR_final_result['3rd Level(Model Prediction)']['Profanity Classification'][0]['Insult'])
            vAR_final_result['3rd Level(Model Prediction)']['Profanity Classification'][0]['Identity Hate'] = str(vAR_final_result['3rd Level(Model Prediction)']['Profanity Classification'][0]['Identity Hate']) 



            vAR_final_result['3rd Level(Model Prediction)']['Sum of all Categories'] = str(vAR_final_result['3rd Level(Model Prediction)']['Sum of all Categories'])

            vAR_final_result['Order Id'] = str(vAR_final_result['Order Id'])

            vAR_final_result['Vehicle Id Number(VIN)'] = str(vAR_final_result['Vehicle Id Number(VIN)'])
            vAR_final_result = json.dumps(vAR_final_result)
            vAR_final_result = json.loads(vAR_final_result)
            if "Error Message" in vAR_final_result.keys():
                print('Below Error in Order Id - '+str(vAR_batch_elp_configuration['ORDER ID'][elp_idx]))
                print(vAR_final_result)
            else:
                print('Order Id - '+str(vAR_batch_elp_configuration['ORDER ID'][elp_idx])+' Successfully processed')
                vAR_response_dict = Process_API_Response(vAR_final_result,vAR_batch_elp_configuration['REQUEST DATE'][elp_idx],vAR_batch_elp_configuration['ORDER DATE'][elp_idx],vAR_batch_elp_configuration['CONFIGURATION'][elp_idx],vAR_batch_elp_configuration['ORDER ID'][elp_idx],vAR_batch_elp_configuration['VIN'][elp_idx],"RNN")
                vAR_output = vAR_output.append(vAR_response_dict,ignore_index=True) 
        vAR_output_copy = vAR_output.copy(deep=True)
        vAR_output = vAR_output.to_csv()
        vAR_request_id = randint(10001, 50000)
        Insert_Response_to_Bigquery(vAR_output_copy,vAR_request_id)
        Upload_Response_To_S3(vAR_output_copy,vAR_request_id)
        return {'Message':'Success','Status':200}
        # elif request_json['model'].upper()=='BERT':

        #     vAR_result,vAR_result_data,vAR_result_target_sum = BERT_Model_Result(vAR_input_text)
        #     vAR_result_data = vAR_result_data.to_json(orient='records')
        #     vAR_random_id = Random_Id_Generator()
        #     if vAR_result_target_sum>20:
        #         vAR_message_level_3 = "Configuration Denied, Since the profanity probability exceeds the threshold"
        #     else:
        #         vAR_message_level_3 = "Level 3 Accepted"
        #     vAR_response_time = round(time.time() - start_time,2)

        #     return {"1st Level(Direct Profanity)":{"Is accepted":not vAR_profanity_result,"Message":vAR_message_level_1},
        #     "2nd Level(Denied Pattern)":{"Is accepted":vAR_regex_result,"Message":vAR_message_level_2},
        #     "3rd Level(Model Prediction)":{"Is accepted":vAR_result,"Message":vAR_message_level_3,"Profanity Classification":json.loads(vAR_result_data),
        #     'Sum of all Categories':vAR_result_target_sum},
        #     'Order Id':vAR_random_id,'Configuration':vAR_input_text,
        #     'Response time':str(vAR_response_time)+" secs","Vehicle Id Number(VIN)":vAR_vin}
        
    except BaseException as e:
        print('In Error Block - '+str(e))
        print(traceback.format_exc())
        return {'Error Message':str(e)}



def Random_Id_Generator():
    vAR_random_id = randint(10001, 50000)
    return vAR_random_id



def LSTM_Model_Result(vAR_input_text):
    # Input Data Preprocessing
    vAR_data = pd.DataFrame()
    vAR_target_columns = ['Toxic','Severe Toxic','Obscene','Threat','Insult','Identity Hate']
    vAR_model_obj = ClassificationModels(vAR_data,vAR_target_columns)
    vAR_test_data = pd.DataFrame([vAR_input_text],columns=['comment_text'])
    vAR_test_data['Toxic'] = None
    vAR_test_data['Severe Toxic'] = None
    vAR_test_data['Obscene'] = None
    vAR_test_data['Threat'] = None
    vAR_test_data['Insult'] = None
    vAR_test_data['Identity Hate'] = None
    print('Xtest length - ',len(vAR_test_data))
    vAR_corpus = vAR_model_obj.data_preprocessing(vAR_test_data)
    print('Data Preprocessing Completed')
    vAR_X,vAR_y = vAR_model_obj.word_embedding_vectorization(vAR_corpus,vAR_test_data)
    print('Vectorization Completed Using Word Embedding')
    print('var X - ',vAR_X)
    print('var Y - ',vAR_y)
    
    vAR_load_model = tf.keras.models.load_model('gs://dmv_elp_project/saved_model/LSTM/LSTM_RNN_Model')

    vAR_model_result = vAR_load_model.predict(vAR_X)
    print('LSTM result - ',vAR_model_result)
    vAR_result_data = pd.DataFrame(vAR_model_result,columns=vAR_target_columns)
    vAR_target_sum = (np.sum(vAR_model_result)*100).round(2)
    vAR_result_data.index = pd.Index(['Percentage'],name='category')
    vAR_result_data = vAR_result_data.astype(float).round(5)*100
    vAR_result_data = vAR_result_data.astype(str)
    print('Inside LSTM Method - ',vAR_result_data)
    # Sum of predicted value with 20% as threshold
    if vAR_target_sum>20:
        return False,vAR_result_data,vAR_target_sum
    else:
        return True,vAR_result_data,vAR_target_sum





def BERT_Model_Result(vAR_input_text):
    
    
    
    vAR_test_sentence = vAR_input_text
    vAR_target_columns = ['Toxic','Severe Toxic','Obscene','Threat','Insult','Identity Hate']
    
    # Name of the BERT model to use
    model_name = 'bert-base-uncased'

    # Max length of tokens
    max_length = 128

    # Load transformers config and set output_hidden_states to False
    config = BertConfig.from_pretrained(model_name)
    #config.output_hidden_states = False

    # Load BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
    
    vAR_test_x = tokenizer(
    text=vAR_test_sentence,
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)
    start_time = time.time()
    # print('Copying Model')
    # subprocess.call(["gsutil cp gs://dsai_saved_models/BERT/model.h5 /tmp/"],shell=True)
    # print('Model File successfully copied')  
    MODEL_PATH = 'gs://dmv_elp_project/saved_model/BERT/model.h5'
    # MODEL_PATH = 'gs://dsai_saved_models/BERT/BERT_MODEL_64B_4e5LR_3E'
    FS = gcsfs.GCSFileSystem()
    with FS.open(MODEL_PATH, 'rb') as model_file:
         model_gcs = h5py.File(model_file, 'r')
         vAR_load_model = tf.keras.models.load_model(model_gcs,compile=False)
    # vAR_load_model = tf.keras.models.load_model('gs://dsai_saved_models/BERT/model.h5',compile=False)
    # vAR_load_model = tf.keras.models.load_model(MODEL_PATH,compile=False)
    # vAR_load_model = tf.keras.models.load_model('/tmp/model.h5',compile=False)

    # vAR_load_model = Load_BERT_Model()
    
    print("---Model loading time %s seconds ---" % (time.time() - start_time))
    

    vAR_model_result = vAR_load_model.predict(x={'input_ids': vAR_test_x['input_ids'], 'attention_mask': vAR_test_x['attention_mask']},batch_size=32)
    
    # if "vAR_load_model" not in st.session_state:
    #     st.session_state.vAR_load_model = tf.keras.models.load_model('DSAI_Model_Implementation_Sourcecode/BERT_MODEL_64B_4e5LR_3E')
    # vAR_model_result = st.session_state.vAR_load_model.predict(x={'input_ids': vAR_test_x['input_ids'], 'attention_mask': vAR_test_x['attention_mask']},batch_size=32)
    vAR_result_data = pd.DataFrame(vAR_model_result,columns=vAR_target_columns)
    vAR_target_sum = (np.sum(vAR_model_result)*100).round(2)
    vAR_result_data.index = pd.Index(['Percentage'],name='category')
    vAR_result_data = vAR_result_data.astype(float).round(5)*100
    
    if vAR_target_sum>20:
        return False,vAR_result_data,vAR_target_sum
    else:
        return True,vAR_result_data,vAR_target_sum




def Number_Replacement(vAR_val):
    vAR_output = vAR_val
    if "1" in vAR_val:
        vAR_output = vAR_output.replace("1","I")
    if "2" in vAR_val:
        vAR_output = vAR_output.replace("2","Z")
    if "3" in vAR_val:
        vAR_output = vAR_output.replace("3","E")
    if "4" in vAR_val:
        vAR_output = vAR_output.replace("4","A")
    if "5" in vAR_val:
        vAR_output = vAR_output.replace("5","S")
    if "8" in vAR_val:
        vAR_output = vAR_output.replace("8","B")
        print('8 replaced with B - ',vAR_val)
    if "0" in vAR_val:
        vAR_output = vAR_output.replace("0","O")
    print('number replace - ',vAR_output)
    return vAR_output



def Binary_Search(data, x):
    low = 0
    high = len(data) - 1
    mid = 0
    i =0
    while low <= high:
        i = i+1
        print('No.of iteration - ',i)
        mid = (high + low) // 2
        
        # If x is greater, ignore left half
        if data[mid] < x:
            low = mid + 1
 
        # If x is smaller, ignore right half
        elif data[mid] > x:
            high = mid - 1
 
        # means x is present at mid
        else:
            return mid
 
    # If we reach here, then the element was not present
    return -1

def Profanity_Words_Check(vAR_val):
    vAR_input = vAR_val
    vAR_badwords_df = pd.read_csv('gs://dmv_elp_project/data/badwords_list.csv',header=None)
    print('data - ',vAR_badwords_df.head(20))
    vAR_result_message = ""
    
#---------------Profanity logic implementation with O(log n) time complexity-------------------
    # Direct profanity check
    vAR_badwords_df[1] = vAR_badwords_df[1].str.upper()
    vAR_is_input_in_profanity_list = Binary_Search(vAR_badwords_df[1],vAR_input)
    if vAR_is_input_in_profanity_list!=-1:
        vAR_result_message = 'Input ' +vAR_val+ ' matches with direct profanity - '+vAR_badwords_df[1][vAR_is_input_in_profanity_list]
        
        return True,vAR_result_message
    
    # Reversal profanity check
    vAR_reverse_input = "".join(reversed(vAR_val)).upper()
    vAR_is_input_in_profanity_list = Binary_Search(vAR_badwords_df[1],vAR_reverse_input)
    if vAR_is_input_in_profanity_list!=-1:
        vAR_result_message = 'Input ' +vAR_val+ ' matches with reversal profanity - '+vAR_badwords_df[1][vAR_is_input_in_profanity_list]
        return True,vAR_result_message
    
    # Number replacement profanity check
    vAR_number_replaced = Number_Replacement(vAR_val).upper()
    vAR_is_input_in_profanity_list = Binary_Search(vAR_badwords_df[1],vAR_number_replaced)
    if vAR_is_input_in_profanity_list!=-1: 
       vAR_result_message = 'Input ' +vAR_val+ ' matches with number replacement profanity - '+vAR_badwords_df[1][vAR_is_input_in_profanity_list]
       return True,vAR_result_message
    
    # Reversal Number replacement profanity check(5sa->as5->ass)
    vAR_number_replaced = Number_Replacement(vAR_reverse_input).upper()
    vAR_is_input_in_profanity_list = Binary_Search(vAR_badwords_df[1],vAR_number_replaced)
    if vAR_is_input_in_profanity_list!=-1:  
        vAR_result_message = 'Input ' +vAR_val+ ' matches with reversal number replacement profanity - '+vAR_badwords_df[1][vAR_is_input_in_profanity_list]
        return True,vAR_result_message
    
    print('1st lvl message - ',vAR_result_message)
    return False,vAR_result_message


def Upload_Request_GCS(vAR_request):
    vAR_request = vAR_request.to_csv()
    vAR_bucket_name = 'dmv_elp_project'
    vAR_bucket = storage.Client().get_bucket(vAR_bucket_name)
    # define a dummy dict
    vAR_utc_time = datetime.datetime.utcnow()
    client = storage.Client()
    bucket = client.get_bucket(vAR_bucket_name)
    vAR_file_path = 'requests/'+vAR_utc_time.strftime('%Y%m%d')+'/dmv_api_request_'+vAR_utc_time.strftime('%H%M%S')+'.csv'
    bucket.blob(vAR_file_path).upload_from_string(vAR_request, 'text/csv')
    print('ELP Configuration Request successfully saved into cloud storage')
    return vAR_file_path



def Insert_Response_to_Bigquery(vAR_df,vAR_request_id):
    vAR_df.rename(columns = {'REQUEST DATE':'REQUEST_DATE','ORDER DATE':'ORDER_DATE','ORDER ID':'ORDER_ID','DIRECT PROFANITY':'DIRECT_PROFANITY',
'DIRECT PROFANITY MESSAGE':'DIRECT_PROFANITY_MESSAGE','RULE-BASED CLASSIFICATION':'RULE_BASED_CLASSIFICATION','RULE-BASED CLASSIFICATION MESSAGE':'RULE_BASED_CLASSIFICATION_MESSAGE','SEVERE TOXIC':'SEVERE_TOXIC','IDENTITY HATE':'IDENTITY_HATE','OVERALL PROBABILITY':'OVERALL_PROBABILITY'
}, inplace = True)
    vAR_request_ids = []
    created_at = []
    created_by = []
    updated_at = []
    updated_by = []
    df_length = len(vAR_df)
    vAR_request_ids += df_length * [vAR_request_id]
    created_at += df_length * [datetime.datetime.utcnow()]
    created_by += df_length * ['Streamlit-User']
    updated_by += df_length * ['']
    updated_at += df_length * ['']
    vAR_df['REQUEST_ID'] = vAR_request_ids
    vAR_df['CREATED_AT'] = created_at
    vAR_df['CREATED_BY'] = created_by
    vAR_df['UPDATED_AT'] = updated_at
    vAR_df['UPDATED_BY'] = updated_by

    # Load client
    client = bigquery.Client(project='elp-2022-352222')

    # Define table name, in format dataset.table_name
    table = 'DMV_ELP_DATASET.DMV_ELP_API_RESULT'
    job_config = bigquery.LoadJobConfig(schema=[bigquery.SchemaField("ORDER_ID", bigquery.enums.SqlTypeNames.INTEGER),bigquery.SchemaField("VIN", bigquery.enums.SqlTypeNames.INTEGER),],write_disposition="WRITE_APPEND",)
    # Load data to BQ
    job = client.load_table_from_dataframe(vAR_df, table,job_config=job_config)

    job.result()  # Wait for the job to complete.
    table_id = 'elp-2022-352222.DMV_ELP_DATASET.DMV_ELP_API_RESULT'
    table = client.get_table(table_id)  # Make an API request.
    print(
        "Loaded {} rows and {} columns to {}".format(
            table.num_rows, len(table.schema), table_id
        )
    )
    print('API Request&Response successfully saved into Bigquery table')


def Process_API_Response(vAR_api_response,vAR_request_date,vAR_order_date,vAR_configuration,vAR_order_id,vAR_vin,vAR_model):
    # vAR_api_response as dict
    vAR_data = {}
    vAR_data['REQUEST DATE'] = vAR_request_date
    vAR_data['ORDER DATE'] = vAR_order_date
    vAR_data['CONFIGURATION'] = vAR_configuration
    vAR_data['ORDER ID'] = vAR_order_id
    vAR_data['VIN'] = vAR_vin
    if vAR_api_response['1st Level(Direct Profanity)']['Is accepted']:
        vAR_data['DIRECT PROFANITY'] = 'APPROVED'
        vAR_data['DIRECT PROFANITY MESSAGE'] = 'Not falls under any of the profanity word'
    if not vAR_api_response['1st Level(Direct Profanity)']['Is accepted']:
        vAR_data['DIRECT PROFANITY'] = 'DENIED'
        vAR_data['DIRECT PROFANITY MESSAGE'] = vAR_api_response['1st Level(Direct Profanity)']['Message']
    if vAR_api_response['2nd Level(Denied Pattern)']['Is accepted']:
        vAR_data['RULE-BASED CLASSIFICATION'] = 'APPROVED'
        vAR_data['RULE-BASED CLASSIFICATION MESSAGE'] = 'Not falls under any of the denied patterns'
    if not vAR_api_response['2nd Level(Denied Pattern)']['Is accepted']:
        vAR_data['RULE-BASED CLASSIFICATION'] = 'DENIED'
        vAR_data['RULE-BASED CLASSIFICATION MESSAGE'] = vAR_api_response['2nd Level(Denied Pattern)']['Message']
    vAR_data['MODEL'] = vAR_model
    vAR_data['TOXIC'] = vAR_api_response['3rd Level(Model Prediction)']['Profanity Classification'][0]['Toxic']
    vAR_data['SEVERE TOXIC'] = vAR_api_response['3rd Level(Model Prediction)']['Profanity Classification'][0]['Severe Toxic']
    vAR_data['OBSCENE'] = vAR_api_response['3rd Level(Model Prediction)']['Profanity Classification'][0]['Obscene']
    vAR_data['IDENTITY HATE'] = vAR_api_response['3rd Level(Model Prediction)']['Profanity Classification'][0]['Identity Hate']
    vAR_data['INSULT'] = vAR_api_response['3rd Level(Model Prediction)']['Profanity Classification'][0]['Insult']
    vAR_data['THREAT'] = vAR_api_response['3rd Level(Model Prediction)']['Profanity Classification'][0]['Threat']
    vAR_data['OVERALL PROBABILITY'] = vAR_api_response['3rd Level(Model Prediction)']['Sum of all Categories']
    
    return vAR_data


def Upload_Response_To_S3(vAR_result,vAR_request_id):
    aws_access_key_id,aws_secret_access_key,vAR_bucket_name = get_gcp_secret()
    vAR_csv_buffer = StringIO()
    vAR_result.to_csv(vAR_csv_buffer)
    vAR_s3_resource = boto3.resource('s3',aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key)
    vAR_utc_time = datetime.datetime.utcnow()
    vAR_s3_resource.Object(vAR_bucket_name, 'batch/simpligov/new/ELP_Project_Response/'+vAR_utc_time.strftime('%Y%m%d')+'/ELP_Response_'+str(vAR_request_id)+'_'+vAR_utc_time.strftime('%H%M%S')+'.csv').put(Body=vAR_csv_buffer.getvalue())
    print('API Response successfully saved into S3 bucket')
    
    
def get_gcp_secret():

    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()
    
    # Build the resource name of the secret version.
    aws_access_key_name =  'projects/elp-2022-352222/secrets/ACCESS_KEY/versions/1'
    aws_secret_key_name =  'projects/elp-2022-352222/secrets/SECRET_KEY/versions/1'
    bucket_name =  'projects/elp-2022-352222/secrets/BUCKET_NAME/versions/1'
    
    # Access the secret version.
    aws_access_key_response = client.access_secret_version(request={"name": aws_access_key_name})
    aws_secret_key_response = client.access_secret_version(request={"name": aws_secret_key_name})
    bucket_name_response = client.access_secret_version(request={"name": bucket_name})
    # WARNING: Do not print the secret in a production environment - this
    # snippet is showing how to access the secret material.
    aws_access_key = aws_access_key_response.payload.data.decode("UTF-8")
    aws_secret_key = aws_secret_key_response.payload.data.decode("UTF-8")
    bucket_name_value = bucket_name_response.payload.data.decode("UTF-8")
    return aws_access_key,aws_secret_key,bucket_name_value