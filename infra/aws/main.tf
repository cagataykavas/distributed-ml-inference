terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" { region = var.aws_region }

data "aws_availability_zones" "available" { state = "available" }

resource "aws_vpc" "ml" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = { Name = "ml-inference-vpc" }
}

resource "aws_internet_gateway" "igw" { vpc_id = aws_vpc.ml.id }

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.ml.id
  cidr_block              = cidrsubnet(aws_vpc.ml.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags = { Tier = "public" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.ml.id
  cidr_block        = cidrsubnet(aws_vpc.ml.cidr_block, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags = { Tier = "private" }
}

resource "aws_eip" "nat" { domain = "vpc" }
resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  depends_on    = [aws_internet_gateway.igw]
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.ml.id
  route { cidr_block = "0.0.0.0/0"; gateway_id = aws_internet_gateway.igw.id }
}
resource "aws_route_table_association" "public" {
  count = 2
  subnet_id = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.ml.id
  route { cidr_block = "0.0.0.0/0"; nat_gateway_id = aws_nat_gateway.nat.id }
}
resource "aws_route_table_association" "private" {
  count = 2
  subnet_id = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

resource "aws_security_group" "alb" {
  name = "inference-alb"
  vpc_id = aws_vpc.ml.id
  ingress { from_port = 443; to_port = 443; protocol = "tcp"; cidr_blocks = ["0.0.0.0/0"] }
  egress { from_port = 0; to_port = 0; protocol = "-1"; cidr_blocks = ["0.0.0.0/0"] }
}

resource "aws_security_group" "service" {
  name = "inference-service"
  vpc_id = aws_vpc.ml.id
  ingress { from_port = 8000; to_port = 8000; protocol = "tcp"; security_groups = [aws_security_group.alb.id] }
  egress { from_port = 0; to_port = 0; protocol = "-1"; cidr_blocks = ["0.0.0.0/0"] }
}

resource "aws_ecr_repository" "model" { name = "distributed-ml-inference" }
resource "aws_ecs_cluster" "main" { name = "ml-inference" }

variable "aws_region" { type = string; default = "eu-central-1" }

output "vpc_id" { value = aws_vpc.ml.id }
output "private_subnets" { value = aws_subnet.private[*].id }
output "ecr_repository_url" { value = aws_ecr_repository.model.repository_url }
