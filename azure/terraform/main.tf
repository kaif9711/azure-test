# --- Provider Configuration ---
terraform {
  required_providers {
    azurerm = { source = "hashicorp/azurerm", version = "~>3.0" }
    random  = { source = "hashicorp/random", version = "~>3.1" }
  }
}

provider "azurerm" {
  features {}
}

# --- Helper to get your Azure Account details ---
data "azurerm_client_config" "current" {}

# --- Naming & Location Setup ---
resource "random_string" "unique" {
  length  = 5
  special = false
  upper   = false
}

locals {
  # Using centralus as it is less restricted
  location      = "centralus" 
  unique_suffix = random_string.unique.result
  
  # Resource Group
  rg_name = "capstone-rg-${local.unique_suffix}"
  
  # Tier 1
  agw_pip_name = "capstone-pip-agw-${local.unique_suffix}"
  agw_name     = "capstone-agw-${local.unique_suffix}"
  
  # Tier 2 - Networking
  vnet_name          = "capstone-vnet-${local.unique_suffix}"
  nat_pip_name       = "capstone-pip-nat-${local.unique_suffix}"
  nat_gw_name        = "capstone-nat-${local.unique_suffix}"
  app_nsg_name       = "capstone-nsg-app-${local.unique_suffix}"
  db_nsg_name        = "capstone-nsg-db-${local.unique_suffix}"
  
  # Tier 2 - Compute & Security
  acr_name           = "capstoneacr${local.unique_suffix}" # No hyphens, global
  kv_name            = "capstone-kv-${local.unique_suffix}" # Global
  aks_name           = "capstone-aks-${local.unique_suffix}"
  identity_name      = "capstone-id-${local.unique_suffix}"
  bastion_pip_name   = "capstone-pip-bas-${local.unique_suffix}"
  bastion_name       = "capstone-bastion-${local.unique_suffix}"
  
  # Tier 2 - Data & Storage
  mysql_name         = "capstone-mysql-${local.unique_suffix}"
  redis_name         = "capstone-redis-${local.unique_suffix}"
  cosmos_name        = "capstone-cosmos-${local.unique_suffix}" # Global
  storage_name       = "capstonest${local.unique_suffix}" # No hyphens, global
  recovery_vault_name = "capstone-rsv-${local.unique_suffix}"
  
  # Tier 3 - Monitoring
  log_analytics_name = "capstone-log-${local.unique_suffix}"
  prometheus_name    = "capstone-monitoring-${local.unique_suffix}"
  grafana_name       = "capstone-graf-${local.unique_suffix}"
}

# --- Main Resource Group ---
resource "azurerm_resource_group" "rg" {
  name     = local.rg_name
  location = local.location
}

# --- Tier 1: Public Edge ---
resource "azurerm_public_ip" "app_gateway_pip" {
  name                = local.agw_pip_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  allocation_method   = "Static"
  sku                 = "Standard"
}

resource "azurerm_application_gateway" "app_gateway" {
  name                = local.agw_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location

  sku {
    name     = "WAF_v2" # Includes Web Application Firewall (WAF)
    tier     = "WAF_v2"
  }
  
  waf_configuration {
    enabled                  = true
    firewall_mode            = "Prevention"
    rule_set_version         = "3.2"
  }
  
  gateway_ip_configuration {
    name      = "app-gateway-ip-config"
    subnet_id = azurerm_subnet.app_gateway_subnet.id
  }
  
  # ... Add frontend_ip_configuration, backend_pools, listeners, etc. ...
  # This part is highly specific to your application's domain name and ports.
}

# --- Tier 2: Private Application & Data Layer ---

# --- Network Security Groups (Firewall Rules) ---
resource "azurerm_network_security_group" "app_nsg" {
  name                = local.app_nsg_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  # ... Add rules to allow traffic from App Gateway ...
}

resource "azurerm_network_security_group" "db_nsg" {
  name                = local.db_nsg_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  # ... Add rules to allow traffic from App Subnet to DB ports ...
}

# --- NAT Gateway (For Outbound Internet from AKS) ---
resource "azurerm_public_ip" "nat_pip" {
  name                = local.nat_pip_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  allocation_method   = "Static"
  sku                 = "Standard"
}

resource "azurerm_nat_gateway" "nat_gateway" {
  name                = local.nat_gw_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku_name            = "Standard"
  public_ip_address_ids = [azurerm_public_ip.nat_pip.id]
}

# --- Networking ---
resource "azurerm_virtual_network" "vnet" {
  name                = local.vnet_name
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

# --- Subnets ---
resource "azurerm_subnet" "app_gateway_subnet" {
  name                 = "App-Gateway-Subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.0.0/24"]
}

resource "azurerm_subnet" "application_subnet" {
  name                 = "Application-Subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.1.0/24"]
  nat_gateway_id       = azurerm_nat_gateway.nat_gateway.id
}

resource "azurerm_subnet_network_security_group_association" "app_nsg_assoc" {
  subnet_id                 = azurerm_subnet.application_subnet.id
  network_security_group_id = azurerm_network_security_group.app_nsg.id
}

resource "azurerm_subnet" "db_subnet" {
  name                 = "DB-Subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.2.0/24"]
  private_endpoint_network_policies_enabled = true
}

resource "azurerm_subnet_network_security_group_association" "db_nsg_assoc" {
  subnet_id                 = azurerm_subnet.db_subnet.id
  network_security_group_id = azurerm_network_security_group.db_nsg.id
}

resource "azurerm_subnet" "bastion_subnet" {
  name                 = "AzureBastionSubnet" # MUST be this exact name
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.3.0/24"]
}

resource "azurerm_subnet" "monitoring_subnet" {
  name                 = "monitoring-subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.4.0/24"]
}

# --- Application & Compute ---
resource "azurerm_container_registry" "acr" {
  name                = local.acr_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Standard" # Production SKU
}

resource "azurerm_user_assigned_identity" "aks_identity" {
  name                = local.identity_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_kubernetes_cluster" "aks" {
  name                = local.aks_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = "capstone-aks"

  default_node_pool {
    name       = "default"
    vm_size    = var.aks_vm_size # Use a variable for production
    node_count = 3             # Start with 3 nodes for production
    vnet_subnet_id = azurerm_subnet.application_subnet.id 
  }

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.aks_identity.id]
  }
  
  managed_aad_profile {
    managed_aad_enabled   = true
    admin_group_object_ids = [var.aks_admin_group_object_id]
  }
}

# --- Security & Access ---
resource "azurerm_key_vault" "keyvault" {
  name                = local.kv_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id
    secret_permissions = [
      "Get", "List", "Set", "Delete", "Purge", "Recover"
    ]
  }
}

resource "azurerm_key_vault_secret" "mysql_admin_password" {
  name         = "mysql-admin-password"
  value        = var.mysql_admin_password
  key_vault_id = azurerm_key_vault.keyvault.id
}

resource "azurerm_public_ip" "bastion_pip" {
  name                = local.bastion_pip_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  allocation_method   = "Static"
  sku                 = "Standard"
}

resource "azurerm_bastion_host" "bastion" {
  name                = local.bastion_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  ip_configuration {
    name                 = "configuration"
    subnet_id            = azurerm_subnet.bastion_subnet.id
    public_ip_address_id = azurerm_public_ip.bastion_pip.id
  }
}

# --- Data & Storage ---
resource "azurerm_mysql_flexible_server" "mysql" {
  name                = local.mysql_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  
  administrator_login    = var.mysql_admin_login
  administrator_password = azurerm_key_vault_secret.mysql_admin_password.value
  sku_name               = var.mysql_sku # Use a variable for production
}

resource "azurerm_redis_cache" "redis" {
  name                = local.redis_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  capacity            = var.redis_capacity # Use a variable for production
  family              = "C"
  sku_name            = "Standard" # Use Standard for production
}

resource "azurerm_cosmosdb_account" "cosmosdb" {
  name                = local.cosmos_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"
  
  consistency_policy {
    consistency_level = "Session"
  }
  
  geo_location {
    location          = azurerm_resource_group.rg.location
    failover_priority = 0
  }
}

resource "azurerm_storage_account" "storage" {
  name                = local.storage_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  account_tier        = "Standard"
  account_replication_type = "LRS"
}

# --- Private DNS Zones (Critical for Private Endpoints) ---
resource "azurerm_private_dns_zone" "mysql_zone" {
  name                = "privatelink.mysql.database.azure.com"
  resource_group_name = azurerm_resource_group.rg.name
}
resource "azurerm_private_dns_zone" "redis_zone" {
  name                = "privatelink.redis.cache.windows.net"
  resource_group_name = azurerm_resource_group.rg.name
}
resource "azurerm_private_dns_zone" "cosmos_zone" {
  name                = "privatelink.documents.azure.com"
  resource_group_name = azurerm_resource_group.rg.name
}
resource "azurerm_private_dns_zone" "storage_zone" {
  name                = "privatelink.blob.core.windows.net"
  resource_group_name = azurerm_resource_group.rg.name
}

# --- Private Endpoints for Data ---
resource "azurerm_private_endpoint" "mysql_pe" {
  name                = "capstone-pep-mysql"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  subnet_id           = azurerm_subnet.db_subnet.id

  private_dns_zone_group {
    name                 = "default"
    private_dns_zone_ids = [azurerm_private_dns_zone.mysql_zone.id]
  }
  private_service_connection {
    name                           = "mysql-psc"
    is_manual_connection           = false
    private_connection_resource_id = azurerm_mysql_flexible_server.mysql.id
    subresource_names              = ["mysqlServer"]
  }
}

resource "azurerm_private_endpoint" "redis_pe" {
  name                = "capstone-pep-redis"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  subnet_id           = azurerm_subnet.db_subnet.id
  private_dns_zone_group {
    name                 = "default"
    private_dns_zone_ids = [azurerm_private_dns_zone.redis_zone.id]
  }
  private_service_connection {
    name                           = "redis-psc"
    is_manual_connection           = false
    private_connection_resource_id = azurerm_redis_cache.redis.id
    subresource_names              = ["redisCache"]
  }
}

resource "azurerm_private_endpoint" "cosmos_pe" {
  name                = "capstone-pep-cosmos"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  subnet_id           = azurerm_subnet.db_subnet.id
  private_dns_zone_group {
    name                 = "default"
    private_dns_zone_ids = [azurerm_private_dns_zone.cosmos_zone.id]
  }
  private_service_connection {
    name                           = "cosmos-psc"
    is_manual_connection           = false
    private_connection_resource_id = azurerm_cosmosdb_account.cosmosdb.id
    subresource_names              = ["Sql"]
  }
}

resource "azurerm_private_endpoint" "storage_pe" {
  name                = "capstone-pep-storage"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  subnet_id           = azurerm_subnet.db_subnet.id
  private_dns_zone_group {
    name                 = "default"
    private_dns_zone_ids = [azurerm_private_dns_zone.storage_zone.id]
  }
  private_service_connection {
    name                           = "storage-psc"
    is_manual_connection           = false
    private_connection_resource_id = azurerm_storage_account.storage.id
    subresource_names              = ["blob"]
  }
}

# --- Backup & Recovery ---
resource "azurerm_recovery_services_vault" "recovery_vault" {
  name                = local.recovery_vault_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "Standard"
}

# --- Tier 3: Monitoring & Analytics ---
resource "azurerm_log_analytics_workspace" "logs" {
  name                = local.log_analytics_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "PerGB2018"
}

resource "azurerm_monitor_workspace" "prometheus" {
  name                = local.prometheus_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
}

resource "azurerm_dashboard_grafana" "grafana" {
  name                = local.grafana_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Standard"
  
  azure_monitor_workspace_integrations {
    resource_id = azurerm_monitor_workspace.prometheus.id
  }
}