# Created by jing at 30.05.23
from aitk import valuation, facts_converter, nsfr


def get_vm(args, lang):
    vm = valuation.get_valuation_module(args, lang)
    return vm


def get_fc(args, lang, VM):
    fc = facts_converter.FactsConverter(args, lang, VM)
    return fc


def get_nsfr(args, lang, FC, train=False):
    NSFR = nsfr.get_nsfr_model(args, lang, FC, train)
    return NSFR
