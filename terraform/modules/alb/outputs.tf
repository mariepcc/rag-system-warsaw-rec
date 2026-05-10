output "alb_dns_name" {
  value       = aws_lb.main.dns_name
  description = "Adres ALB np. spotguide-alb.eu-north-1.elb.amazonaws.com - potrzebny dla modułu dns_records"
}

output "target_group_arn" {
  value       = aws_lb_target_group.api.arn
  description = "ARN target group — ECS rejestruje tu swoje kontenery"
}

output "alb_security_group_id" {
  value       = aws_security_group.alb.id
  description = "Security group ALB — ECS musi przyjmować ruch tylko z ALB"
}

output "alb_listener_https_arn" {
  value = aws_lb_listener.https.arn
}

output "alb_zone_id" {
  description = "Zone ID ALB — potrzebny dla modułu dns_records"
  value       = aws_lb.main.zone_id
}

output "alb_arn_suffix" {
  value = aws_lb.main.arn_suffix
}

output "target_group_arn_suffix" {
  value = aws_lb_target_group.api.arn_suffix
}