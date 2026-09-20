"""
Email Service for DataDuck 2MFA OTP Verification
"""
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


def generate_otp_code() -> str:
    """Generate a secure 6-digit numeric OTP code."""
    return f"{secrets.randbelow(1000000):06d}"


async def send_otp_email(to_email: str, otp_code: str, purpose: str = "verification") -> bool:
    """
    Send OTP verification or password reset email.
    If SMTP server settings are configured, sends via SMTP.
    Otherwise, logs the OTP prominently to the server output for development.
    """
    is_reset = purpose == "password_reset"
    subject_title = "Password Reset Code" if is_reset else "Verification Code"
    heading_text = "Reset Your Password" if is_reset else "Verify Your Account"
    desc_text = (
        "Enter the following 6-digit code to reset your DataDuck account password."
        if is_reset
        else "Enter the following 6-digit code to complete registration on DataDuck."
    )

    # Always log to dev console for clear testing visibility
    logger.info("==================================================")
    logger.info(f"🔑 [{subject_title.upper()}] Sent to: {to_email}")
    logger.info(f"👉 YOUR OTP CODE IS: {otp_code}")
    logger.info("==================================================")

    smtp_host = getattr(settings, "SMTP_HOST")
    smtp_port = getattr(settings, "SMTP_PORT")
    smtp_user = getattr(settings, "SMTP_USER")
    smtp_password = getattr(settings, "SMTP_PASSWORD")

    if smtp_host and smtp_user and smtp_password:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"Your DataDuck {subject_title}: {otp_code}"
            
            # Format sender with custom app display name: "DataDuck <email@domain.com>"
            sender_email = getattr(settings, "EMAILS_FROM", None) or smtp_user
            msg["From"] = formataddr(("DataDuck", sender_email))
            msg["To"] = to_email

            text_content = f"Your {subject_title.lower()} for DataDuck (Doubt. Dig. Discover.) is: {otp_code}\nThis code will expire in 10 minutes."
            html_content = f"""
            <div style="font-family: 'Helvetica Neue', Arial, sans-serif; max-width: 500px; margin: 0 auto; padding: 30px; background-color: #0d0d0d; color: #ffffff; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                <h2 style="color: #ffffff; text-align: center; margin-bottom: 4px;">DataDuck</h2>
                <p style="color: #6b7280; text-align: center; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px;">Doubt. Dig. Discover.</p>
                <h3 style="color: #ffffff; text-align: center; margin-bottom: 8px;">{heading_text}</h3>
                <p style="color: #9ca3af; text-align: center; font-size: 14px; margin-bottom: 24px;">{desc_text}</p>
                <div style="background: rgba(255,255,255,0.05); border: 1px dashed rgba(255,255,255,0.2); border-radius: 8px; padding: 20px; text-align: center; margin-bottom: 24px;">
                    <span style="font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #10b981;">{otp_code}</span>
                </div>
                <p style="color: #6b7280; text-align: center; font-size: 12px;">This code expires in 10 minutes. If you did not request this code, please ignore this email.</p>
            </div>
            """
            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(msg["From"], [to_email], msg.as_string())
            
            logger.info(f"Successfully sent {purpose} email via SMTP to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}. (Logged OTP code to console above)")
            return False
    
    return True
