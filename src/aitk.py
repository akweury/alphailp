# Created by shaji on 21-Apr-23

import percept
import valuation
import facts_converter


def get_vm(args, lang):
    vm = valuation.get_valuation_module(args, lang)
    return vm


def get_pm(args):
    pm = percept.get_perception_module(args)
    return pm


def get_fc(args, lang, VM):
    fc = facts_converter.FactsConverter(args, lang, VM)
    return fc


def get_pi_vm(args, lang):
    pi_vm = valuation.PIValuationModule(lang=lang, device=args.device, dataset=args.dataset,
                                        dataset_type=args.dataset_type)
    return pi_vm

def get_nsfr():

    return
