
## Parser Utils
def replace_module(model, old_module, replacer, module_name = ''):
    # Recursively replace modules in the copied model
    for name, module in model.named_children():
        full_name = f'{module_name}.{name}' if module_name != '' else name
        if not hasattr(module, 'nvdla'):
            if isinstance(module, old_module):
                if hasattr(module, 'groups'):
                    if (module.groups > 1):
                        # Bypass DW layers
                        return model
                # Instantiate the replacementV module
                replacement = replacer(module, full_name)
                setattr(model, name, replacement)
            else:
                # If the module has children, apply recursively
                replace_module(module, old_module, replacer, full_name)
    
    return model  # Return the modified model copy

def replace_singleModule(model, old_module_name, supported_list, replacers_list, module_name = ''):
    # Recursively replace modules in the copied model
    for name, module in model.named_children():
        full_name = f'{module_name}.{name}' if module_name != '' else name
        if not hasattr(module, 'nvdla'):
            # if isinstance(module, old_module):
            if full_name == old_module_name:
                # Instantiate the replacement module
                for idx, supported_module in enumerate(supported_list):
                    if isinstance(module, supported_module):
                        replacer = replacers_list[idx]
                        replacement = replacer(module, full_name)
                        setattr(model, name, replacement)
            #     replacement = replacer(module, full_name)
            #     setattr(model, name, replacement)
            else:
                # If the module has children, apply recursively
                replace_singleModule(module, old_module_name, supported_list, replacers_list, full_name)
    
    return model  # Return the modified model copy
