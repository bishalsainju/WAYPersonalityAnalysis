import re

def processStatus(status):
        status = status.lower()
        status = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', status)
        status = re.sub('@[^\s]+', 'AT_USER', status)
        status = re.sub('[\s]+', ' ', status)
        status = status.strip('\'"')
        return status
