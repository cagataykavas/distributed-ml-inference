terraform {
  required_version = ">= 1.6.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  availability_zone_count = 2
}

resource "aws_vpc" "ml" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = "ml-inference-vpc" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.ml.id
}

resource "aws_subnet" "public" {
  count                   = local.availability_zone_count
  vpc_id                  = aws_vpc.ml.id
  cidr_block              = cidrsubnet(aws_vpc.ml.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags                    = { Tier = "public" }
}

resource "aws_subnet" "private" {
  count             = local.availability_zone_count
  vpc_id            = aws_vpc.ml.id
  cidr_block        = cidrsubnet(aws_vpc.ml.cidr_block, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags              = { Tier = "private" }
}

resource "aws_eip" "nat" {
  count  = local.availability_zone_count
  domain = "vpc"
  tags   = { Name = "ml-inference-nat-${count.index}" }
}

resource "aws_nat_gateway" "nat" {
  count         = local.availability_zone_count
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  depends_on    = [aws_internet_gateway.igw]
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.ml.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
}

resource "aws_route_table_association" "public" {
  count          = local.availability_zone_count
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  count  = local.availability_zone_count
  vpc_id = aws_vpc.ml.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat[count.index].id
  }
}

resource "aws_route_table_association" "private" {
  count          = local.availability_zone_count
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

resource "aws_security_group" "alb" {
  name   = "inference-alb"
  vpc_id = aws_vpc.ml.id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "service" {
  name   = "inference-service"
  vpc_id = aws_vpc.ml.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecr_repository" "model" {
  name = "distributed-ml-inference"
}

resource "aws_ecs_cluster" "main" {
  name = "ml-inference"
}

variable "aws_region" {
  type    = string
  default = "eu-central-1"
}

output "vpc_id" {
  value = aws_vpc.ml.id
}

output "private_subnets" {
  value = aws_subnet.private[*].id
}

output "nat_gateway_ids" {
  value       = aws_nat_gateway.nat[*].id
  description = "One NAT Gateway per private-subnet Availability Zone."
}

output "ecr_repository_url" {
  value = aws_ecr_repository.model.repository_url
}
