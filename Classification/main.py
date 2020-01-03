from options import get_parsed_args
import os
import pprint as pp
from datasets import dataloaders_factory
from options import  get_parsed_args, parser
from models import create_feature_extractor,




def main(args, trainer):
    pp.pprint(vars(args), width=1)

    ###############
    #gpu setting하고, 환경 setting 하는 것 나중에 추가하기
    ###############
    # export_root, args = _setup_experiments(args)

    # data 부분 세련되게 하는 것은 나중에 보완하기
    dataloaders = dataloaders_factory(args)

    feature_extractor = create_feature_extractor(args)
    #classifier = create_class_classifier(args)





if __name__ == "__main__":
    parsed_args =  get_parsed_args(parser)
    main(parsed_args, trainer) #trainer?