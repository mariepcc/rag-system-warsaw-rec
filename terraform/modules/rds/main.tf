resource "aws_db_subnet_group" "main" {
  name       = "spotguide-db-subnet-private"
  subnet_ids = var.subnet_ids
}

resource "aws_security_group" "rds" {
  name   = "spotguide-rds-sg"
  vpc_id = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    cidr_blocks = [var.vpc_cidr]  
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "main" {
  identifier        = "spotguide-db"
  engine            = "postgres"
  engine_version    = "15"
  instance_class    = "db.t4g.micro"  
  allocated_storage = 20

  db_name  = "spotguide"
  username = "spotguide_user"
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.main.name
  apply_immediately      = true
  vpc_security_group_ids = [aws_security_group.rds.id]
  
  publicly_accessible     = false  
  skip_final_snapshot     = true   
  backup_retention_period = 7
  backup_window           = "03:00-04:00" 
  maintenance_window      = "mon:04:00-mon:05:00"      
  multi_az                = false  
  tags = { Name = "spotguide-db" }
}