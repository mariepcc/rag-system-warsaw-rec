terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket = "spotguide-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "eu-north-1"
  }
}

provider "aws" {
  region = var.aws_region
}

module "networking" {
  source = "./modules/networking"
}

module "ecr" {
  source          = "./modules/ecr"
  repository_name = var.app_name
  tags = {
    Environment = var.environment
    Project     = var.app_name
  }
}

module "cognito" {
  source       = "./modules/cognito"
  project_name = var.app_name
  environment  = var.environment
}

module "rds" {
  source      = "./modules/rds"
  vpc_id      = module.networking.vpc_id
  subnet_ids  = module.networking.private_subnet_ids
  db_password = var.db_password
}

module "dns" {
  source      = "./modules/dns"
  domain_name = var.domain_name
}

module "ssl" {
  source      = "./modules/ssl"
  domain_name = var.domain_name
  zone_id     = module.dns.zone_id
}

module "alb" {
  source          = "./modules/alb"
  vpc_id          = module.networking.vpc_id
  subnet_ids      = module.networking.public_subnet_ids
  certificate_arn = module.ssl.certificate_arn
}

module "dns_records" {
  source       = "./modules/dns_records"
  zone_id      = module.dns.zone_id
  domain_name  = var.domain_name
  alb_dns_name = module.alb.alb_dns_name
  alb_zone_id  = module.alb.alb_zone_id
}

module "ecs" {
  source                 = "./modules/ecs"
  vpc_id                 = module.networking.vpc_id
  subnet_ids             = module.networking.public_subnet_ids
  app_image              = "${module.ecr.repository_url}:latest"
  openai_api_key         = var.openai_api_key
  cognito_user_pool_id   = module.cognito.user_pool_id
  cognito_app_client_id  = module.cognito.app_client_id
  database_url           = "postgresql://${var.db_username}:${var.db_password}@${module.rds.db_endpoint}/warsaw_places"
  alb_security_group_id  = module.alb.alb_security_group_id
  target_group_arn       = module.alb.target_group_arn
  alb_https_listener_arn = module.alb.alb_listener_https_arn
}

module "monitoring" {
  source = "./modules/monitoring"

  ecs_cluster_name       = "spotguide-cluster"
  ecs_service_name       = "spotguide-api"
  db_instance_identifier = "spotguide-db"
  alb_arn_suffix         = module.alb.alb_arn_suffix
  target_group_arn_suffix = module.alb.target_group_arn_suffix
}

module "ga" {
  source = "./modules/ga"

  app_name        = var.app_name
  github_username = var.github_username
  github_repo     = var.github_repo
}