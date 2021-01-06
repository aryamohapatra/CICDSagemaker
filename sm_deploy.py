import sagemaker
import boto3
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch import PyTorchModel
from comment import Comment

sagemaker_session = sagemaker.Session(boto3.session.Session())

# Put the right role and input data
role = "arn:aws:iam::869082236477:role/service-role/AmazonSageMaker-ExecutionRole-20201125T135948"



trainedmodel = sagemaker.model.Model(
    model_data='s3://sagex-pipeline-data/SagePoc1-335b438-2021-01-06-09-05-31/output/model.tar.gz', 
    image = '811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest',
    role=role)  # your role here; could be different name

#comment = Comment()
#values = comment.get_comment('model_data=')
#if values is None or len(values) == 0:
 #   comment.add_comment('Deploy Fail: no model data. Did you train?')
  #  exit(-1)

#print("Data:", values[-1])

#model = PyTorchModel(model_data=values[-1],
 #                    role=role,
  #                   framework_version='1.5.0',
   #                  entry_point='mnist.py',
    #                 source_dir='code')


#comment.add_comment('Deploying with data ' + values[-1])
try:
    predictor = trainedmodel.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')
    print('end_point=' + predictor.endpoint)
except Exception as e:
    print('Deploy Fail:' + str(e))
