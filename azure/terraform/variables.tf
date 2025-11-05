variable "mysql_admin_login" {
  description = "Admin username for the MySQL server (no hyphens)"
  type        = string
  sensitive   = true 
}

variable "mysql_admin_password" {
  description = "Admin password for the MySQL server"
  type        = string
  sensitive   = true
}

variable "aks_admin_group_object_id" {
  description = "The Object ID of the Entra ID (Azure AD) group for AKS admins"
  type        = string
}

variable "aks_vm_size" {
  description = "The VM size for the AKS node pool"
  type        = string
  default     = "Standard_DS2_v2"
}

variable "mysql_sku" {
  description = "The SKU for the MySQL flexible server"
  type        = string
  default     = "B_Standard_B2s" # A good default production 'Burstable' tier
}

variable "redis_capacity" {
  description = "The size for the Standard Redis cache (C1 = 1, C2 = 2, etc.)"
  type        = number
  default     = 1 # C1 (Standard)
}