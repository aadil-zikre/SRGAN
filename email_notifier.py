import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import sys
import traceback

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import sys
import traceback

class EmailNotifier:
    def __init__(self, smtp_server, smtp_port, sender_email, sender_password):
        """
        Initialize the EmailNotifier instance with SMTP server details and sender credentials.

        Args:
            smtp_server (str): SMTP server address. # smtp.gmail.com for gmail.
            smtp_port (int): SMTP server port. # For SSL, enter 465. For TLS, enter 587.
            sender_email (str): Sender's email address.
            sender_password (str): Sender's email password.
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.error_recipients = None

    def send_email(self, subject, body, recipients, cc=None, bcc=None, attachments=None):
        """
        Send an email with the provided subject, body, recipients, CC, BCC, and attachments.

        Args:
            subject (str): Email subject.
            body (str): Email body.
            recipients (str or list): Email recipients. Can be a single email or a list of emails.
            cc (str or list, optional): CC recipients. Can be a single email or a list of emails. Defaults to None.
            bcc (str or list, optional): BCC recipients. Can be a single email or a list of emails. Defaults to None.
            attachments (list, optional): List of file paths to attach. Defaults to None.
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['Subject'] = subject

            if isinstance(recipients, list):
                msg['To'] = ', '.join(recipients)
            else:
                msg['To'] = recipients

            if cc:
                if isinstance(cc, list):
                    msg['CC'] = ', '.join(cc)
                else:
                    msg['CC'] = cc

            if bcc:
                if isinstance(bcc, list):
                    msg['BCC'] = ', '.join(bcc)
                else:
                    msg['BCC'] = bcc

            msg.attach(MIMEText(body, 'plain'))

            if attachments:
                for attachment in attachments:
                    with open(attachment, 'rb') as file:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(file.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', f'attachment; filename={attachment}')
                        msg.attach(part)

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            print("Email sent successfully!")
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            self.send_error_email(str(e), traceback.format_exc())

    def send_error_email(self, error_message, traceback_info):
        """
        Send an error notification email with the provided error message and traceback information.

        Args:
            error_message (str): Error message.
            traceback_info (str): Traceback information.
        """
        subject = f"Script Failure Notification: {self.script_name}" if self.script_name else "Script Failure Notification"
        body = f"The script encountered an error:\n\n{error_message}\n\nTraceback:\n{traceback_info}"
        recipients = self.error_recipients or self.sender_email
        self.send_email(subject, body, recipients)

    def set_error_trigger(self, script_name, recipients=None):
        """
        Set the global exception handler to the `global_exception_handler` method of the EmailNotifier instance.

        Args:
            recipients (str or list, optional): Email recipients for error notifications.
                Can be a single email or a list of emails. Defaults to None.
        """
        self.script_name=script_name
        self.error_recipients = recipients
        sys.excepthook = self.global_exception_handler

    def global_exception_handler(self, exctype, value, tb):
        """
        Global exception handler that sends an error notification email and exits the script with a non-zero status code.

        Args:
            exctype: Exception type.
            value: Exception value.
            tb: Traceback object.
        """
        self.send_error_email(str(value), ''.join(traceback.format_exception(exctype, value, tb)))
        print(f"An error occurred: {str(value)}")
        print("Traceback:")
        traceback.print_exception(exctype, value, tb)
        sys.exit(1)