import re
from calendar import month_name


def enlarge_entity_list(entities):
    for key in entities.keys():
        entities_to_expand = entities[key]
        reg_list = []
        for reg in entities_to_expand:
            reg_list.append(reg)
            if '/' in reg:
                reg_list.append(reg.replace('/', ' '))
                reg_list.append(reg.replace('/', '-'))

            if len(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', reg)) > 0 and not is_date(reg):
                str_list = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', reg)
                join_str = ' '.join(str_list)
                entity_type = re.findall(r'\([^)]*\)', reg)
                reg_list.append(join_str + entity_type[0])

            if '.java' in str(reg):
                reg_list.append(reg.replace('.java', ' .java'))
                reg_list.append(reg.replace('.java', '. java'))
        entities[key] = reg_list
    return entities


def label_entities(dataset, entities):
    new_dataset = {}
    for i in dataset.keys():
        labeled_entitiy_list = []
        for j in dataset[i]:
            # labelling
            for r in entities[i]:
                cleaned_r = re.sub(r'\([^)]*\)', '', r)
                type = re.sub(r'\[[^)]*\]', '', r)
                cleaned_r = cleaned_r.replace('[', '').replace(']', '')
                matched_entities = re.findall(cleaned_r, j, flags=re.IGNORECASE)
                for entity in matched_entities:
                    if type == entity:
                        continue
                    j = j.replace(entity, "[" + entity + "]" + type)
            labeled_entitiy_list.append(j)
        new_dataset[i] = labeled_entitiy_list
    return new_dataset


def is_date(word):
    month_master = ('jan', 'feb', 'march', 'april', 'may', 'june', 'july', 'august', 'sept', 'oct', 'nov', 'dec')
    for m in month_name:
        if str.lower(m) in str.lower(word):
            return True

    for m in month_master:
        if m in str.lower(word):
            return True
    return False
