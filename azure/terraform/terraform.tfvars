# --- terraform.tfvars ---
# [!!] FILL IN YOUR SECRETS HERE [!!]

# --- MySQL Credentials ---
# (Remember: no hyphens in the login name)
mysql_admin_login    = "capstone_admin"
mysql_admin_password = "Muskaan-123!"

# --- AKS Admin Group ---
# (Get this from Azure Portal > Microsoft Entra ID > Groups > Your Group > Object Id)
aks_admin_group_object_id = "1acf45dc-9b5f-427d-86b0-d024de5d25ab"

# --- Production Sizing (Optional) ---
# (You can override the defaults from variables.tf here if you want)
#
# aks_vm_size  = "Standard_D4s_v3"
# mysql_sku    = "GP_Standard_D2ds_v4"
# redis_capacity = 2