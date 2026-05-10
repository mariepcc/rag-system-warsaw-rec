variable "ecs_cluster_name" {
  type = string
}

variable "ecs_service_name" {
  type = string
}

variable "db_instance_identifier" {
  type = string
}

variable "alb_arn_suffix" {
  type        = string
  description = "Suffix ARN ALB — z outputs modułu alb"
}

variable "target_group_arn_suffix" {
  type        = string
  description = "Suffix ARN Target Group — z outputs modułu alb"
}