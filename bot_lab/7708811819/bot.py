import random
import uuid
import asyncio
import json
import os
import time
import logging
import difflib
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import dns.resolver

# Fix for Termux/Android missing /etc/resolv.conf
try:
    dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
    dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']
except:
    pass

from pymongo import MongoClient
from telegram.request import HTTPXRequest
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand, InputMediaPhoto, InputMediaVideo, ReplyKeyboardMarkup, ReplyKeyboardRemove, constants
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

# Global dictionary to keep player data in RAM
player_cache = {}
BUTTON_COOLDOWNS = {}
RARITY_STYLES = {
    "Common": {"symbol": "🔘", "label": "🔘 Common"},
    "Rare": {"symbol": "🔮", "label": "🔮 Rare"},
    "Legendary": {"symbol": "⚜️", "label": "⚜️ Legendary"}
}

RIDDLES = [
    # --- BASICS ---
    {"hint": "the object to stop a ship 🛑", "correct": "⚓️", "options": ["⚔️", "⚓️", "🧭"]},
    {"hint": "the weapon of a true swordsman 🤺", "correct": "⚔️", "options": ["🏹", "⚔️", "🛡"]},
    {"hint": "what you need to steer the ship ☸️", "correct": "☸️", "options": ["🛶", "⚓️", "☸️"]},
    {"hint": "the Jolly Roger flag 🏴‍☠️", "correct": "🏴‍☠️", "options": ["🚩", "🏳️", "🏴‍☠️"]},
    {"hint": "used to find treasure 🗺️", "correct": "🗺️", "options": ["📜", "🗺️", "🔭"]},
    {"hint": "used to spot land from afar 🔭", "correct": "🔭", "options": ["🔭", "🔫", "🕯️"]},
    
    # --- ONE PIECE LORE ---
    {"hint": "the fruit that gives powers 🍇", "correct": "😈", "options": ["🍎", "😈", "🍌"]},
    {"hint": "the currency of the seas 💰", "correct": "🍇", "options": ["🍇", "💵", "💎"]},
    {"hint": "Luffy's favorite food 🍖", "correct": "🍖", "options": ["🍜", "🍖", "🍙"]},
    {"hint": "Zoro's drink of choice 🍶", "correct": "🍶", "options": ["🥛", "🍶", "🍵"]},
    {"hint": "Nami's favorite fruit 🍊", "correct": "🍊", "options": ["🍊", "🍒", "🍑"]},
    {"hint": "Sanji's weapon (his legs) 🦵", "correct": "🦵", "options": ["👊", "🦵", "🗡️"]},
    {"hint": "Chopper's favorite sweet 🍬", "correct": "🍬", "options": ["🍬", "🍰", "🍫"]},
    {"hint": "Franky's fuel source 🥤", "correct": "🥤", "options": ["⛽", "🥤", "☕"]},
    
    # --- COMBAT & ITEMS ---
    {"hint": "protects you from attacks 🛡️", "correct": "🛡️", "options": ["🛡️", "⚔️", "🧶"]},
    {"hint": "fires explosive balls 💣", "correct": "💣", "options": ["🎱", "💣", "🏺"]},
    {"hint": "a sniper's best friend 🎯", "correct": "🏹", "options": ["🏹", "🎣", "🦯"]},
    {"hint": "the Log Pose compass 🧭", "correct": "🧭", "options": ["⌚", "🧭", "⏲️"]},
    {"hint": "a Marine ship 🛳️", "correct": "🛳️", "options": ["🛳️", "⛵", "🛶"]},
    {"hint": "the treasure chest 📦", "correct": "📦", "options": ["📦", "📪", "🧱"]}
]

ADMIN_IDS = [5242138546, 7708811819]
# =====================
# LOGGING SETUP
# =====================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# =====================
# DATA PERSISTENCE (MONGODB ATLAS)
# =====================

MONGO_URI = "mongodb+srv://Znxpirateshowdown:1234Qwer%3F%3F@znx.idxdehh.mongodb.net/?appName=Znx"

try:
    if MONGO_URI:
        mongo_client = MongoClient(
            MONGO_URI, 
            serverSelectionTimeoutMS=5000, 
            maxPoolSize=50, 
            retryWrites=True
        )
        mongo_client.admin.command('ping')
        db = mongo_client["pirate_v3"]
        players_collection = db["players"]
        print("✅ Connected to MongoDB Atlas successfully.")
    else:
        print("⚠️ MONGO_URI not found in environment.")
        mongo_client = None
        players_collection = None
except Exception as e:
    print(f"❌ Failed to connect to MongoDB: {e}")
    mongo_client = None
    players_collection = None

def init_db():
    if mongo_client:
        try:
            db = mongo_client["pirate_v3"]
            db["players"].create_index("user_id", unique=True)
            db["players"].create_index("bounty", direction=-1) # For Leaderboard
            print("✅ Database index created/verified.")
        except Exception as e:
            print(f"Database connection warning: {e}")

def load_player(user_id):
    uid = str(user_id)
    if uid in player_cache:
        p = player_cache[uid]
        if "characters" in p:
            for c in p["characters"]:
                if c.get("level", 1) > 40:
                    c["level"] = 40
        return p
        
    if players_collection is None: 
        return None
        
    try:
        data = players_collection.find_one({"user_id": uid}, {"_id": 0})
        
        if data:
            updated = False
            # Fix legacy fields
            if "is_locked" not in data: data["is_locked"] = False; updated = True
            if "verification_active" not in data: data["verification_active"] = False; updated = True
            if "artifacts" not in data: data["artifacts"] = []; updated = True
            if "keys" not in data: data["keys"] = []; updated = True
            if "bounty" not in data: data["bounty"] = 100000; updated = True
            
            # Daily Task Fields
            if "daily_stats" not in data:
                data["daily_stats"] = {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "kills": 0, "bounty_gained": 0, "clovers_gained": 0,
                    "claimed": False
                }
                updated = True
            
            # Enforce Level 40 Cap on DB Load
            if "characters" in data:
                for c in data["characters"]:
                    if c.get("level", 1) > 40:
                        c["level"] = 40
                        updated = True

            if updated:
                players_collection.update_one({"user_id": uid}, {"$set": data})

            player_cache[uid] = data 
            return data
            
        return data
    except Exception as e:
        logging.error(f"Error loading player {user_id}: {e}")
        return None


def save_player(user_id, player_data):
    uid = str(user_id)
    player_cache[uid] = player_data
    
    if players_collection is None: return
    try:
        players_collection.update_one(
            {"user_id": uid},
            {"$set": player_data},
            upsert=True
        )
    except Exception as e:
        logging.error(f"Error saving player {user_id}: {e}")


init_db()

# =====================
# CONSTANTS & STATS
# =====================

WHEEL_VIDEO = "BAACAgUAAyEFAATl0UgqAAIeZWmFStczJlfo5LnlJVRzeWuUtoSLAAJtIAACMx4pVKMk-1-9BG_3OAQ"
YAMATO_ULT_VIDEO = "BAACAgUAAxkBAAIr1WmAKRpP7UEXoxx58xwDRtc65mzSAALQGQACW_z5V441GyK7BGOqOAQ"
KID_ULT_VIDEO = "BAACAgUAAxkBAAIuemmCBZpyHB8s96nGwTuTrPhrqeDlAAKYIwACukcJVL9MKAIltY9HOAQ"
SUMMON_ANIMATION = "BAACAgUAAyEFAATl0UgqAAIWWGmDReT_fHUD07PqTFA7r1TrK_PUAAIJIQACAeUZVObA9U5rAk3jOAQ"

KID_SUMMON_ANIM = "BAACAgUAAxkBAAIxV2mKEHo_X_pEYR0aQr-NHEMP7q38AAKaIAACryVRVFV9W5Uh5TDHOgQ"
YAMATO_SUMMON_ANIM = "BAACAgUAAyEFAATl0UgqAAImqWmKD-7KRrw-sso1fflrKeMwybCIAAKfIAACryVRVLi7jhi0hkHkOgQ"
LAW_SUMMON_ANIM = "BAACAgUAAyEFAAToYlLgAAKBJGmQBWydv9PfCiVqRklrHhf9y_r-AALmFwACY8qAVAWK6y2ITDcROgQ"
ACE_SUMMON_ANIM = "BAACAgUAAyEFAATl0UgqAAI-qmmPgsvPAd2q-AHFRnc4aoEs5isEAALjFwACY8qAVEwWej9Ox86AOgQ"

KID_EXPLORE_ULT = "AgACAgUAAxkBAAIxXWmKEK3AQvUWEetFepCwwCKkFTyrAAKnDmsbukcJVEAQZ7R1nCaCAQADAgADeQADOgQ"
YAMATO_EXPLORE_ULT = "AgACAgUAAxkBAAIxW2mKEKa3uIN7-5Yta1jxPFtZrOFbAALBDWsbW_z5VwQFGVqA2BWRAQADAgADeAADOgQ"

INVENTORY_IMAGE = "AgACAgUAAxkBAAIuz2mDX1GCg9NPgBG0UPXp6zuwsuUjAAIMEGsbCfwYVE0skEm56DPBAQADAgADeQADOAQ"
STORE_IMG = "AgACAgUAAxkBAAIvsmmEYjJo8rnkBkUIxxFbfnBr0I0bAAKmEGsbMx4pVDtFcs56hQohAQADAgADeQADOAQ"

# Chests
DARK_CHEST_IMG = "AgACAgUAAyEFAATl0UgqAAIaemmEQz9KfL_beaSK3YHjzdZrIDppAAJqEGsbMx4pVH2Pdkr37vjJAQADAgADeAADOAQ"
GOLD_CHEST_IMG = "AgACAgUAAyEFAATl0UgqAAIaf2mEQ0NHcjbHdZb1dWsqhZJ4sXKwAAJuEGsbMx4pVM8P8sb8dtXNAQADAgADeAADOAQ"
FROST_CHEST_IMG = "AgACAgUAAyEFAATl0UgqAAIaimmEQ0YhFqdVd5Zq3re9x0bD6F5CAAJxEGsbMx4pVPkAASeeheyLwgEAAwIAA3gAAzgE"

# New Artifact Images
BLUE_ARTIFACT_IMG = "AgACAgUAAxkBAAK17GmRhZ8lX_Jw-4Z_PA-S5t9y5_9dAALnD2sbEsmIVHlL31Wf7pIcAQADAgADeAADOgQ"
PURPLE_ARTIFACT_IMG = "AgACAgUAAxkBAAK17mmRhaPD-4ThHPKZJffwWQyfp4ZyAALoD2sbEsmIVLnHMRE2K7mzAQADAgADeAADOgQ"

NAME_ALIASES = {
    "kid": "Eustass Kid",
    "law": "Trafalgar D. Law",
    "ace": "Portgas D Ace"
}

ULT_IMAGES = {
    "Alvida": "AgACAgUAAxkBAAIwvmmFaNMraidlErRzZgi1TWMPVr2sAAK_DWsbwLX4VyXUp5J3sHNxAQADAgADeQADOAQ",
    "Chopper": "AgACAgUAAxkBAAIwwGmFaRNHwyApKmXmKY1Ag0SJhhTHAAIIFGsbRLwpVOiLpE244XGLAQADAgADeQADOAQ",
    "Arlong": "AgACAgUAAxkBAAIwwmmFakbAtQgSLy8nfirVa28hqJ2zAAISFGsbRLwpVFeRRVKsvFY4AQADAgADeQADOAQ",
    "Nami": "AgACAgUAAxkBAAIwxGmFakwzyAlUYx9Z2jtKSFhuPa-_AAIUFGsbRLwpVLd3FmJH0uDDAQADAgADeQADOAQ",
    "Helmeppo": "AgACAgUAAxkBAAIwxmmFal1aHInnNIhlV7DV1ztYJhkwAAIVFGsbRLwpVJFzZewcCLqcAQADAgADeQADOAQ",
    "Buggy": "AgACAgUAAxkBAAIwyGmFaorEl41lmrnu_yiHEZj6UevDAAIWFGsbRLwpVLlNPNKDVSb9AQADAgADeQADOAQ",
    "Usopp": "AgACAgUAAxkBAAIwymmFaqLzk4NzAAEvHJtBBhij5O_FzgACGBRrG0S8KVR8F-_",
    "Koby": "AgACAgUAAxkBAAIsAmmAp65JZUNHpXF2Fv3FjxeT1MhPAAILDWsbY4oBVKYcdcYZUyUxAQADAgADeAADOAQ"
}

DEVIL_FRUITS = {
    "Sand Sand Fruit": {
        "text": "Sand sand fruit \n\nRarity:⭐️\n\nDevil fruit info: This fruit allow user to manipulate, control and create sand at will. \n\n     Fruits stats\nDefense:32-38\nDamage: 15-25\nCritical chance:25%\nAccuracy:94%\nRank requirement: 7",
        "img": "AgACAgUAAyEFAATl0UgqAAIdCGmEyAP0N7D46htpr52YyW4gk59hAAIuEmsbMx4pVO7ttlPyKjeNAQADAgADeQADOAQ",
        "atk_buff": 20, "def_buff": 35, "hp_buff": 0, "cost": 15000, "lvl": 7
    },
    "Shadow Shadow Fruit": {
        "text": "Shadow shadow fruit \n\nRarity:⭐️\n\nDevil fruit info: This fruit allows users to manipulate, manifest, and steal shadow and turn into corpse. \n\n     Fruits stats\nDamage: 30-32\nDefense: 15-20\nCritical chance: 20%\nAccuracy: 91%\nRank requirement: 4",
        "img": "AgACAgUAAyEFAATl0UgqAAIdFWmE7Liugh6UWYR8q5tA_sHCPNsdAAIvEmsbMx4pVEpP7t_9_F97AQADAgADeAADOAQ",
        "atk_buff": 31, "def_buff": 17, "hp_buff": 0, "cost": 15000, "lvl": 4
    },
    "Barrier Barrier Fruit": {
        "text": "Barrier barrier fruit \n\nRarity:⭐️\n\nDevil fruit info: This fruit allows consumer to turn any weapon or surrounding into strong barrier. \n\n     Fruits stats\nDefense: 25-35\nAttack:5-10\nCritical chance: 15%\nAccuracy: 98%\nRank requirement:1",
        "img": "AgACAgUAAyEFAATl0UgqAAIeemmE7IhCyP3Xv5z0MWur0XyybbDSAAJxEmsbMx4pVCqTaJaSZ0K8AQADAgADeQADOAQ",
        "atk_buff": 7, "def_buff": 30, "hp_buff": 0, "cost": 10000, "lvl": 1
    },
    "Munch Munch Fruit": {
        "text": "Munch munch fruit \n\nRarity:⭐️\n\nDevil fruit info: This fruit allows consume any substance and incorporate into their body. \n\n     Fruits stats\nHP: 30-40\nDamage: 15-20\nCritical chance: 12%\nAccuracy: 95%\nRank requirement: 1",
        "img": "AgACAgUAAyEFAATl0UgqAAIemWmE90tMDiO7XXEHktwrz9tFu_V9AALLEmsbMx4pVCKf_jfVPhlrAQADAgADeQADOAQ",
        "atk_buff": 17, "def_buff": 0, "hp_buff": 35, "cost": 10000, "lvl": 1
    },
    "Gum Gum Fruit": {
        "text": "Gum gum fruit\n\nRarity:⭐️\n\nDevil fruit info: This fruit turns consumer's body into rubber, allowing them to stretch.\n\n     Fruits stats\nDamage: 25-30\nDefense: 10-20\nCritical chance: 10%\nAccuracy: 90%\nRank Requirement: 1",
        "img": "AgACAgUAAyEFAATl0UgqAAIecmmE7I5fGowjyIYU7_df7Dwxf1UyAAJkEmsbMx4pVHs0wiHNY7NzAQADAgADeQADOAQ",
        "atk_buff": 27, "def_buff": 15, "hp_buff": 0, "cost": 10000, "lvl": 1
    }
}

WEAPONS = {
    "Dual Katana": {
        "rarity": "⭐️", "atk_range": "35-40", "atk_val": 45, "crit": "10%", "acc": "98%", "spec": "Dual slash", "lvl": 1, "cost": 10000,
        "img": "AgACAgUAAyEFAATl0UgqAAIZt2mEOqRXXmoQl-ulHSOIrVWLQjzoAAJUFmsbMx4hVLLn3TFeaW8CAQADAgADeAADOAQ"
    },
    "Triple Katana": {
        "rarity": "⭐️", "atk_range": "45-50", "atk_val": 55, "crit": "10%", "acc": "98%", "spec": "Triple Tornado", "lvl": 1, "cost": 20000,
        "img": "AgACAgUAAyEFAATl0UgqAAIZuWmEOqlKixV19PQIvi96-GuoPHIKAAJWFmsbMx4hVHbg-8QGvEdRAQADAgADeAADOAQ"
    },
    "Shark Saw": {
        "rarity": "⭐️", "atk_range": "50-55", "atk_val": 60, "crit": "15%", "acc": "98%", "spec": "Shark resonance", "lvl": 1, "cost": 25000,
        "img": "AgACAgUAAyEFAATl0UgqAAIZu2mEOq10xpnpB49zbKaEg-j40GqKAAJXFmsbMx4hVFAid74FSZU-AQADAgADeAADOAQ"
    },
    "Green Blade": {
        "rarity": "⭐️", "atk_range": "60-70", "atk_val": 75, "crit": "25%", "acc": "98%", "spec": "Green slash", "lvl": 5, "cost": 45000,
        "img": "AgACAgUAAyEFAATl0UgqAAIZw2mEOrMJjjEtehNfXfiSBoJxMCBiAAJYFmsbMx4hVIch9ahbbQQMAQADAgADeQADOAQ"
    },
    "Magma Dagger": {
        "rarity": "⭐️", "atk_range": "80-90", "atk_val": 100, "crit": "25%", "acc": "95%", "spec": "Magma Force", "lvl": 15, "cost": 65000,
        "img": "AgACAgUAAyEFAATl0UgqAAIZ7WmEOry82XEfPl1oUYc0_KU4djqAAAJZFmsbMx4hVGmQ1rgUOcjsAQADAgADeQADOAQ"
    },
    "Azure Needle": {
        "rarity": "⭐️⭐️", "atk_range": "100-110", "atk_val": 125, "crit": "30%", "acc": "95%", "spec": "Azure Counter", "lvl": 30, "cost": 70000,
        "img": "AgACAgUAAyEFAATl0UgqAAIaAAFphDzjTunksxUIOZYWj3FlV9BURwACWhZrGzMeIVQ0u2UqGs_0JgEAAwIAA3kAAzgE"
    },
    "Forest Blade": {
        "rarity": "⭐️⭐️", "atk_range": "130-150", "atk_val": 160, "crit": "35%", "acc": "95%", "spec": "Forest god slash", "lvl": 30, "cost": 85000,
        "img": "AgACAgUAAyEFAATl0UgqAAIaFWmEPQJkm-U2-AlHDexL0Ke8vLYpAAJcFmsbMx4hVCdm7R62bN3iAQADAgADeQADOAQ"
    }
}

SELL_PRICES = {
    "Blue-Artifact": 25000,
    "Purple-Artifact": 50000
}

BOSS_MISSIONS = {
    15: {"name": "Arlong", "img": "AgACAgUAAxkBAAIs9mmAsORj03tw4HZ2sKKGwEms-wu7AAJyDGsb19YJVFX3zXQ6I9cxAQADAgADeAADOAQ", "mission_num": 1},
    30: {"name": "Piccolo", "img": "AgACAgUAAxkBAAIs-WmAsQgNDT_G4xg1HsGZuxkcdFnNAAJzDGsb19YJVIQu9AK0gu4dAQADAgADeQADOAQ", "mission_num": 2},
    50: {"name": "Rui", "img": "AgACAgUAAxkBAAIs_GmAsU-6zEng4yccNa3jO4gvmZREAAJ1DGsb19YJVH82a_F3nMnQAQADAgADeQADOAQ", "mission_num": 3},
    100: {"name": "Crocodile", "img": "AgACAgUAAxkBAAIs_2mAsYd_L-nQWCy3hg5LJtYpljZeAAJ2DGsb19YJVGU6G-b2-XALAQADAgADeQADOAQ", "mission_num": 4},
    150: {"name": "Itachi Uchiha", "img": "AgACAgUAAxkBAAItAmmAsafdewloIC8XlfjrJH9pe9aOAAJ3DGsb19YJVCgdljzoYSuWAQADAgADeQADOAQ", "mission_num": 5},
    175: {"name": "Feitan Portan", "img": "AgACAgUAAxkBAAItBWmAsdjhWowAAeqczS1z10GvXLwhcwACeAxrG9fWCVSbcVihfLwy4AEAAwIAA3gAAzgE", "mission_num": 6},
    200: {"name": "Cell", "img": "AgACAgUAAxkBAAItCGmAsgr3LLepNQABIZVvHeGIOvWmpwACeQxrG9fWCVQmfj_9ateLEQEAAwIAA3kAAzgE", "mission_num": 7},
    250: {"name": "Stark", "img": "AgACAgUAAxkBAAItC2mAskg-PoSoVlUo5Qgc-8uvjQhRAAJ6DGsb19YJVKsAAWIESVmIKgEAAwIAA3kAAzgE", "mission_num": 8},
    300: {"name": "Broly", "img": "AgACAgUAAxkBAAItDmmAssckahqYZhiN0uynWs_seJv_AAJ8DGsb19YJVF95gLcEjQYsAQADAgADeQADOAQ", "mission_num": 9},
    350: {"name": "Frieza", "img": "AgACAgUAAxkBAAIsEWmAswleMjJ_SfE8fNFVUqY0CJS3AAJ-DGsb19YJVKnYLmtuOcZkAQADAgADeQADOAQ", "mission_num": 10},
    375: {"name": "Daki", "img": "AgACAgUAAxkBAAItFWmAuR96mNKa6nWKcjtHBfKubbjLAAKiDGsb19YJVPFkdMsNCLzYAQADAgADeAADOAQ", "mission_num": 11},
    400: {"name": "Gyutaro", "img": "AgACAgUAAxkBAAItGGmAuTMCpNCQAx0vb9ZgIueO97wIAAKjDGsb19YJVLihQcGB8TQeAQADAgADeQADOAQ", "mission_num": 12},
    450: {"name": "Dabi", "img": "AgACAgUAAxkBAAItG2mAuTzySUl34WPt97W9FClXI1P4AAKkDGsb19YJVMKVj4qnzg_FAQADAgADeQADOAQ", "mission_num": 13},
    475: {"name": "Blackbeard", "img": "AgACAgUAAxkBAAIs-WmAsQgNDT_G4xg1HsGZuxkcdFnNAAJzDGsb19YJVIQu9AK0gu4dAQADAgADe4dAQADAgADeQADOAQ", "mission_num": 14},
    500: {"name": "Kakashi Hatake", "img": "AgACAgUAAxkBAAItIWmAuVXkpvD5uainre7pr8SjFhS5AAKmDGsb19YJVGAO_rS_wuDOAQADAgADeQADOAQ", "mission_num": 15},
    550: {"name": "Geto", "img": "AgACAgUAAxkBAAItJGmAuV5MgGjeuvA9WtkZp4EfXn6dAAKnDGsb19YJVKd9qK8QNZTHAQADAgADeQADOAQ", "mission_num": 16},
    600: {"name": "Frieren", "img": "AgACAgUAAxkBAAItJ2mAuWkzkL-DCBH3BzmVXJivvfRqAAKoDGsb19YJVGys0QQbelRsAQADAgADeAADOAQ", "mission_num": 17},
    650: {"name": "Black Goku", "img": "AgACAgUAAxkBAAItKmmAuXJYD_h4faJP09TW1job5zPRAAKpDGsb19YJVIv-zCCKuXwjAQADAgADeAADOAQ", "mission_num": 18},
    700: {"name": "Mahito", "img": "AgACAgUAAxkBAAItLWmAuXh1H9IdpBzSD9n10UuOoI5lAAKqDGsb19YJVLhG2SytKJIWAQADAgADeAADOAQ", "mission_num": 19},
    750: {"name": "Yuji Itadori", "img": "AgACAgUAAxkBAAItMGmAuYECU_BK5Gt18HAyz0Jm7WdRAAKrDGsb19YJVJsz3N7U80ABAQADAgADeQADOAQ", "mission_num": 20}
}

EFFECT_DESCRIPTIONS = {
    "Alvida": "Increases defense by 10%.",
    "Chopper": "deals 70 damage and increases every teammate and himself health❤️ by 50 hp.",
    "Arlong": "Deals 85 damage. Increase his attack ⚔by 15% for 2 moves.",
    "Koby": "Deals 70 Damage and increases his chance to dodge next move by 30%.",
    "Usopp": "Deals 75 damage and reduced enemy Defense 🛡by 5%. And heals himself hy 25 Hp for 2 moves .",
    "Buggy": "Deals 80 damage increase all teamates attack⚔ by 5%.",
    "Helmeppo": "Deals 70 Damage and increases his chance to dodge next move by 50%. Increases his teamates speed⚡️by 10%.",
    "Nami": "Deals 70 Damage and stuns💤 enemy for 1 round.",
    "Yamato": "Deals 130 damage. Increases her chances of dodge by 50%. For 2 rounds her attack⚔ increases by 10%. Defense 🛡increases by 15%.",
    "Eustass Kid": "Deals 145 Damage. For 2 rounds Kid increases his attack by 25%. Speed increased by 10%.",
    "Portgas D Ace": "Deals 160 Damage. Increases Speed⚡ by 40% and 30%. Reduces enemy Defense🛡 by 30%.",
    "Trafalgar D. Law": "Deals 150 Damage. Increases Speed⚡ by 40% and Defense🛡 by 30%. Reduces enemy Attack⚔ by 15%."
}

MOVES = {
    "Kanabo smash": {"dmg": 50}, "Slip Slip punch": {"dmg": 55}, "Sube sube no mi": {"dmg": 60, "effect": "def_buff_10"},
    "Heavy gong": {"dmg": 45}, "Kung fu point": {"dmg": 45}, "Kokutei Roseo Metal": {"dmg": 70, "effect": "team_heal_50"},
    "Shark teeth": {"dmg": 65}, "Shark on dart": {"dmg": 70}, "Kiribachi": {"dmg": 85, "effect": "atk_buff_15_2"},
    "Kamisoro": {"dmg": 40}, "Tempest Kick": {"dmg": 50}, "Honesty impact": {"dmg": 70, "effect": "dodge_30"},
    "Skull Bomb grass": {"dmg": 40}, "Impact wolf": {"dmg": 45}, "Usopp hammer": {"dmg": 75, "effect": "usopp_ult"},
    "Chop Chop canon": {"dmg": 60}, "Chop Chop buzzsaw": {"dmg": 65}, "Bara Bara festival": {"dmg": 80, "effect": "team_atk_5"},
    "Sword swing": {"dmg": 50}, "Dual Kukri": {"dmg": 55}, "Firey morale": {"dmg": 70, "effect": "helmeppo_ult"},
    "Thunderbolt Tempo": {"dmg": 50}, "Swing Arm": {"dmg": 40}, "Zeus breeze tempo": {"dmg": 70, "effect": "stun_1"},
    "Namuji Hyoga": {"dmg": 80}, "Namuji glacier fang": {"dmg": 75}, "Thunder Bagua": {"dmg": 130, "effect": "yamato_ult"},
    "Riperu": {"dmg": 70}, "Punk Gibson": {"dmg": 80}, "Damned Punk": {"dmg": 145, "effect": "kid_ult"},
    "Strike": {"dmg": 30}, "Bash": {"dmg": 35}, "Special Beam": {"dmg": 45}, "Quick Slash": {"dmg": 35}, "Heavy Blow": {"dmg": 40},
    "Dual slash": {"dmg": 45}, "Triple Tornado": {"dmg": 55}, "Shark resonance": {"dmg": 60}, "Green slash": {"dmg": 75},
    "Magma Force": {"dmg": 100}, "Azure Counter": {"dmg": 125}, "Forest god slash": {"dmg": 160},
    # ACE
    "Fire : Gun": {"dmg": 100}, "Entei": {"dmg": 160, "effect": "ace_ult"},
    # LAW
    "Room : Shambles": {"dmg": 100}, "Gamma Knife": {"dmg": 150, "effect": "law_ult"}
}

CHARACTERS = {
    "Alvida": {"rarity": "Common", "class": "Tank🛡", "hp": 600, "atk_min": 22, "atk_max": 22, "def": 30, "spe": 30, "moves": ["Kanabo smash"], "ult": "Sube sube no mi"},
    "Chopper": {"rarity": "Rare", "class": "Healer🧚‍♂", "hp": 700, "atk_min": 30, "atk_max": 35, "def": 40, "spe": 25, "moves": ["Heavy gong"], "ult": "Kokutei Roseo Metal"},
    "Arlong": {"rarity": "Rare", "class": "Damage dealer⚔", "hp": 660, "atk_min": 40, "atk_max": 45, "def": 30, "spe": 35, "moves": ["Shark teeth"], "ult": "Kiribachi"},
    "Koby": {"rarity": "Common", "class": "Assassin 🥷", "hp": 550, "atk_min": 25, "atk_max": 25, "def": 20, "spe": 35, "moves": ["Kamisoro"], "ult": "Honesty impact"},
    "Usopp": {"rarity": "Rare", "class": "Healer 🧚‍♂", "hp": 650, "atk_min": 35, "atk_max": 40, "def": 40, "spe": 30, "moves": ["Skull Bomb grass"], "ult": "Usopp hammer"},
    "Buggy": {"rarity": "Rare", "class": "Damage dealer ⚔", "hp": 620, "atk_min": 40, "atk_max": 45, "def": 25, "spe": 35, "moves": ["Chop Chop canon"], "ult": "Bara Bara festival"},
    "Helmeppo": {"rarity": "Rare", "class": "Assassin 🥷", "hp": 680, "atk_min": 35, "atk_max": 35, "def": 30, "spe": 45, "moves": ["Sword swing"], "ult": "Firey morale"},
    "Nami": {"rarity": "Rare", "class": "Support💪", "hp": 600, "atk_min": 25, "atk_max": 30, "def": 35, "spe": 25, "moves": ["Thunderbolt Tempo"], "ult": "Zeus breeze tempo"},
    "Yamato": {"rarity": "Legendary", "class": "Assassin", "hp": 900, "atk_min": 50, "atk_max": 60, "def": 60, "spe": 50, "moves": ["Namuji Hyoga"], "ult": "Thunder Bagua"},
    "Eustass Kid": {"rarity": "Legendary", "class": "Damage dealer⚔", "hp": 850, "atk_min": 60, "atk_max": 70, "def": 65, "spe": 40, "moves": ["Riperu"], "ult": "Damned Punk"},
    "Portgas D Ace": {"rarity": "Legendary", "class": "Damage Dealer⚔️", "hp": 830, "atk_min": 70, "atk_max": 120, "def": 70, "spe": 55, "moves": ["Fire : Gun"], "ult": "Entei"},
    "Trafalgar D. Law": {"rarity": "Legendary", "class": "SUPPORT 💪", "hp": 840, "atk_min": 80, "atk_max": 100, "def": 100, "spe": 70, "moves": ["Room : Shambles"], "ult": "Gamma Knife"}
}

EXPLORE_DATA = {
    "King": "AgACAgUAAxkBAAIr6mmAp2EYS4XrDKMXRRsowyQ3gfWuAALTDmsbW_wBVCZuP_JtZpU6AQADAgADeAADOAQ",
    "Rob Lucci": "AgACAgUAAxkBAAIr7GmAp2dGkUu9U2zDjJENaIbEhEXdAALYDmsbW_wBVDCtsBlLQUE9AQADAgADeAADOAQ",
    "Black Maria": "AgACAgUAAxkBAAIr7mmAp2yhp6b-TPl_kZoC9Sx_ip7JAALaDmsbW_wBVJBjLXZqyYS4AQADAgADeAADOAQ",
    "Arlong NPC": "AgACAgUAAxkBAAIr8GmAp3FoGJG0zdvL9Fs4qGd-iprHAALbDmsbW_wBVPQ_PZ-g2wjwAQADAgADeAADOAQ",
    "Douglas Bullet": "AgACAgUAAxkBAAIr8mmAp3bMU6vUWJxSI0r6q4nm8r3hAALdDmsbW_wBVM3rxFBI3zKVAQADAgADeAADOAQ",
    "Don krieg": "AgACAgUAAxkBAAIr9GmAp3xMoMASkWzhbhPCtp3T7aaZAALhDmsbW_wBVDa02bAREgkPAQADAgADeAADOAQ",
    "Kuro": "AgACAgUAAxkBAAIr9mmAp4DcGKfugO7-_tM2NBwCiEN6AAKEDmsbCU4BVCEiRHdTM7RgAQADAgADeQADOAQ",
    "Kalifa": "AgACAgUAAxkBAAIr-GmAp4Ubq2XZfqzGQV2qfdqPb8OiAAKFDmsbCU4BVLrZYr2v1n_1AQADAgADeAADOAQ",
    "Ulti": "AgACAgUAAxkBAAIr-mmAp4q7nE5gPuA2i4K6UQo1qbAbAAKGDmsbCU4BVPQqFYrKr717AQADAgADeAADOAQ",
    "NPC Pirate": "AgACAgUAAxkBAAIr_GmAp5tSMZpbfYPU3VoGqodY398MAAJZDWsbW_z5V9-rjIYJ8FDzAQADAgADeQADOAQ",
    "Monet": "AgACAgUAAxkBAAIr_mmAp6NagP0JJ_AsUdJoVdGkDvLkAAIJDWsbY4oBVNAe98Ggvic5AQADAgADeAADOAQ",
    "Doflamingo": "AgACAgUAAxkBAAIsAAFpgKeoyYY2fgwMtvIm2DqtunrdKgACCg1rG2OKAVRNYd_PuO7I6AEAAwIAA3gAAzgE",
    "Smoker": "AgACAgUAAxkBAAIsAmmAp65JZUNHpXF2Fv3FjxeT1MhPAAILDWsbY4oBVKYcdcYZUyUxAQADAgADeAADOAQ",
    "Enel": "AgACAgUAAxkBAAIsBGmAp7NGBf0jobEyAnpSmhPfL3VvAAIMDWsbY4oBVGBVP8GMQgESAQADAgADeAADOAQ",
    "Buggy Clown": "AgACAgUAAxkBAAIsBmmAp7kDSA-CX8RZx1HhbPi0r6jtAAINDWsbY4oBVOiWq2rQ4TFNAQADAgADeAADOAQ",
    "Crocodile": "AgACAgUAAxkBAAIsCGmAp8Cgyg_O5D4s-_S8a19pbu3EAAIZDWsbY4oBVBP3GgiVf8lBAQADAgADeAADOAQ",
    "Pell": "AgACAgUAAxkBAAIsCmmAp8RfZlOnqGJxdJWHlx664LWMAALDDWsb0oUBVAnJaPV02zFcAQADAgADeAADOAQ",
    "Perona": "AgACAgUAAxkBAAIsDGmAp8mypupnW3pKXlMnRrCWyN3hAALIDWsb0oUBVLW6hccH2dyxAQADAgADeAADOAQ",
    "Brook": "AgACAgUAAxkBAAIsDmmAp89uGTJnc65Jkf5e9ro2svqYAALKDWsb0oUBVFeK4PzUUNmTAQADAgADeAADOAQ",
    "Portgas D Ace": "AgACAgUAAxkBAAIsEGmAp9NYcj5JLk75ww3138FuwtKdAALLDWsb0oUBVC4rPNaY3hSNAQADAgADeAADOAQ",
    "Killer": "AgACAgUAAxkBAAIsEmmAp9jVH7jyYecJIx09flxCGinlAALMDWsb0oUBVIn_3xMoF3-6AQADAgADeAADOAQ",
    "Nico Robin": "AgACAgUAAxkBAAIsFGmAp9529v6Di1chuw4_9cfU-EkiAALNDWsb0oUBVKHQ34AWzcxxAQADAgADeQADOAQ",
    "Chopper NPC": "AgACAgUAAxkBAAIsFmmAp-Ps0-kNAAG6wMynROiPP7Kz1wAC0Q1rG9KFAVTfJ8Q_AVxyAQEAAwIAA3gAAzgE",
    "Nami NPC": "AgACAgUAAxkBAAIsGGmAp-hyS4PhRQlgGnMqTk--c_vLAALODWsb0oUBVNeTsJ6uHBbvAQADAgADeQADOAQ",
    "Sabo": "AgACAgUAAxkBAAIsGmmAp-0Scu1YFauEeVHHCLRRt1C4AALSDWsb0oUBVE2FpBZskDHEAQADAgADeAADOAQ",
    "Rosinante": "AgACAgUAAxkBAAIsHGmAp_NvBL1yg_LIAAHjE2B1Y1GNrAAC0w1rG9KFAVRj_yXuH6ZzHgEAAwIAA3gAAzgE",
    "Trafalgar Law": "AgACAgUAAxkBAAIsHmmAp_cnj4ldcMUQb5P_YJUr0sw6AALUDWsb0oUBVBGlM1j0jRV3AQADAgADeAADOAQ",
    "Doll": "AgACAgUAAxkBAAIsIGmAqAEPwN4oGsKGAUvnBrkU-YCvAALVDWsb0oUBVFo-Q7a6rnUPAQADAgADeQADOAQ",
    "Katakuri": "AgACAgUAAxkBAAIsImmAqAVnfjSguZUhUjTwFiEeykTKAALXDWsb0oUBVPuwnTNl2Lw8AQADAgADeAADOAQ",
    "Franky": "AgACAgUAAxkBAAIsJGmAqApwyGDjojhoBFwb59zn2u3gAALYDWsb0oUBVMQkXqlz5vEiAQADAgADeAADOAQ",
    "Senor Pink": "AgACAgUAAxkBAAIsJmmAqA8LC6RISSpb5joQLGN3ivGoAALZDWsb0oUBVFQWgjNjVFn8AQADAgADeAADOAQ",
    "S-Hawk": "AgACAgUAAxkBAAIsKGmAqBQgv3Pi-0vDP1qeW-Q0bE5_AALaDWsb0oUBVPY1sSKf2NPEAQADAgADeAADOAQ",
    "S-Snake": "AgACAgUAAxkBAAIsKmmAqBmqHPO5HNEGsG7F34tcRqARAALbDWsb0oUBVMJs_uiK4tb-AQADAgADeAADOAQ",
    "Pica": "AgACAgUAAxkBAAIsLGmAqB5gB0FNagtLz6K8mUFCYIxdAALfDWsb0oUBVFkBi-8jSNjSAQADAgADeAADOAQ",
    "Jinbe": "AgACAgUAAxkBAAIsLmmAqCPNoAJ9BITlc4BCFS2aFRjSAALgDWsb0oUBVLx6jvmtf0SDAQADAgADeAADOAQ",
    "Nefertari Cobra": "AgACAgUAAxkBAAIsMGmAqClnnxJ6HBQLowHbytwJKhRxAALhDWsb0oUBVIo-1t6SqGnNAQADAgADeAADOAQ",
    "Usopp NPC": "AgACAgUAAxkBAAIsMmmAqC4KFJ5hJ0O8mdk6nsE1vHrOAALjDWsb0oUBVO7B5sGyU4nuAQADAgADeAADOAQ",
    "Daz Bones": "AgACAgUAAxkBAAIsNGmAqDNUHlUfRw1sOZdjCrQBobBsAALkDWsb0oUBVNpgENZvy-4EAQADAgADeAADOAQ",
    "Pedro": "AgACAgUAAxkBAAIsNmmAqDhxEyD3lHufXU64YuZj_o5qAALlDWsb0oUBVDMZN23hyW_QAQADAgADeQADOAQ",
    "Sasaki": "AgACAgUAAxkBAAIsOGmAqDw6P3NWTizNj4jrd8O1YXKcAALnDWsb0oUBVC5iYxJKoamJAQADAgADeAADOAQ",
    "Dellinger": "AgACAgUAAxkBAAIsOmmAqEItemsDUkwiboWcvtMTHwbrTB-AALoDWsb0oUBVMjxgygfT7OSAQADAgADeAADOAQ",
    "Wiper": "AgACAgUAAxkBAAIsPmmAqE9pCvTsRZYTEQNKH84ix3eFAALpDWsb0oUBVHyd9E7yshXsAQADAgADeAADOAQ",
    "Vinsmoke Judge": "AgACAgUAAxkBAAIsQGmAqFb4YMOn4SMWwt-7QWUSqLTJAALqDWsb0oUBVDjzzbqqKxPUAQADAgADeAADOAQ",
    "Kyros": "AgACAgUAAxkBAAIsQmmAqFt3qSkIYPLgUz5V4eQYQAFrAALrDWsb0oUBVHVHMfwF5eYIAQADAgADeAADOAQ",
    "Shiki": "AgACAgUAAxkBAAIsRGmAqGArETkXJBXmPN5zGzx_cs_8AALsDWsb0oUBVHoEYRuKBPOjAQADAgADeAADOAQ",
    "Saint Charlos": "AgACAgUAAxkBAAIsRmmAqGU67kcFgFevliEdrlk9p1XbAALtDWsb0oUBVONdflaMUG2EAQADAgADeAADOAQ",
    "Akainu": "AgACAgUAAxkBAAIsSGmAqHVcbU6TsRT9ZQJ0AdDGAAHPUQAC7w1rG9KFAVRh7peu8gRcCwEAAwIAA3gAAzgE",
    "Apoo": "AgACAgUAAxkBAAIsSmmAqHtJaLc8iYKWyCraXO3ENfROAALwDWsb0oUBVG7jsMjhVDOpAQADAgADeAADOAQ",
    "Boa Hancock": "AgACAgUAAxkBAAIsTGmAqIDcR7y0YE4XM8sAAcVudfgepQAC8Q1rG9KFAVRcM1V6s4E-wgEAAwIAA3gAAzgE",
    "Sugar": "AgACAgUAAxkBAAIsTmmAqIRCQ-YtpDEFm79e8TYoZOq5AALyDWsb0oUBVI2a6eWzpFQiAQADAgADeAADOAQ",
    "Gecko Moria": "AgACAgUAAxkBAAIsUGmAqIqV-24xs8tOeP05aOsE80UHAALzDWsb0oUBVOq8YeCJMa48AQADAgADeAADOAQ",
    "Magellan": "AgACAgUAAxkBAAIsUmmAqI9rzBPliOMSBB3R_E1USh8gAAL0DWsb0oUBVOY9-PIYyp5fAQADAgADeAADOAQ",
    "Koby NPC": "AgACAgUAAxkBAAIsVGmAqJVXMyfs9T20Mxz2jxodNUmTAAL1DWsb0oUBVJoLYOQVzqhNAQADAgADeAADOAQ",
    "Bartholomew Kuma": "AgACAgUAAxkBAAIsVmmAqJp3eBuAona1ASfMCE9SGs5hAAL2DWsb0oUBVN4wVyP6muPLAQADAgADeAADOAQ",
    "Bonney": "AgACAgUAAxkBAAIsWGmAqJ_yk-bvV5J-wWu4DVLLpcXSAAL3DWsb0oUBVC4NxOQyuavxAQADAgADeQADOAQ",
    "Stussy": "AgACAgUAAxkBAAIsWmmAqKX0a4UAAfXc7Fr8VgABdL4b4iAAAvoNaxvShQFUqHnTd24J--0BAAMCAAN5AAM4BA",
    "Lilith": "AgACAgUAAxkBAAIsXGmAqKq11oExlC3h0eoPidbt9PVwAAL8DWsb0oUBVCm1B2-tXE5vAQADAgADeAADOAQ",
    "Nico Olivia": "AgACAgUAAxkBAAIsXmmAqK_CWqp7go5HAAGCOSkiW9q5YAAC_Q1rG9KFAVQIRRg3q0WmkQEAAwIAA3gAAzgE",
    "Caesar Clown": "AgACAgUAAxkBAAIsYGmAqLVVWbEeE1vdt8LlwUrNPIZPAAL-DWsb0oUBVNyKLKFDq2PjAQADAgADeAADOAQ",
    "Jack": "AgACAgUAAxkBAAIsYmmAqMGHno-NAwi8hg9jIjyjW6VZAAL_DWsb0oUBVLeDCEvCFd3SAQADAgADeAADOAQ",
    "Vergo": "AgACAgUAAxkBAAIsZGmAqMZwXhduDyYSwNMZeaylDnHQAAICDmsb0oUBVOcAAZIvcrhHMAEAAwIAA3gAAzgE",
    "Van Augur": "AgACAgUAAxkBAAIsZmmAqM02g4dx-CrtzpXM2oIoPyVlAAIDDmsb0oUBVHBl15mNK7N9AQADAgADeAADOAQ",
    "Helmeppo NPC": "AgACAgUAAxkBAAIsaGmAqNgXfXKfeXfJ1J4sUIzn2lmrAAIEDmsb0oUBVO-gt3HoPNs8AQADAgADeQADOAQ",
    "Emet": "AgACAgUAAxkBAAIsammAqNxVgxkjE1dorKSY4Jxcl7dtAAIGDmsb0oUBVBc9pkn3eeqGAQADAgADeAADOAQ",
    "Hiyori Kozuki": "AgACAgUAAxkBAAIsbGmAqORWYIYM6geIR_ZrY6ti1LzWAAIQDmsb0oUBVIelBAWHq4OlAQADAgADeAADOAQ",
    "Paragus": "AgACAgUAAxkBAAIsbmmAqPPMfsqqRxioGNqX-YltVysbAAJ-Dmsb0oUBVASnsjVwKkW1AQADAgADeAADOAQ",
    "King Vegeta": "AgACAgUAAxkBAAIscGmAqPifpo3L5EjHh4hfNHzQuA-XAAJ_Dmsb0oUBVIdw2lC3kap0AQADAgADeAADOAQ",
    "Android 16": "AgACAgUAAxkBAAIscmmAqP1aDvg0A583aNKkdDVI5wqyAAKADmsb0oUBVBg9cnU6EgcrAQADAgADeAADOAQ",
    "Nappa": "AgACAgUAAxkBAAIsdGmAqQELDMA5AiKK9311da4BAkMGAAKDDmsb0oUBVEN9BzwqV7WNAQADAgADeAADOAQ",
    "Raditz": "AgACAgUAAxkBAAIsdmmAqQfvXHv0LqUCUIFADr74miV-AAKGDmsb0oUBVK-nRknGbJIgAQADAgADeAADOAQ",
    "Android 19": "AgACAgUAAxkBAAIseGmAqQyyYjhzzIOzErfvB7LotoMZAAKHDmsb0oUBVBtPDZGHk4AaAQADAgADeQADOAQ",
    "Zarbon": "AgACAgUAAxkBAAIsemmAqRi60eHtcRmAOYSExMyV_6YGAAKJDmsb0oUBVHS2DrS-RiNnAQADAgADeAADOAQ",
    "Yamcha": "AgACAgUAAxkBAAIr2mmAWGUiHmGXiJ12ZUievoQ9yNPwAAKLDmsb0oUBVK37xzJaQ8hvAQADAgADeAADOAQ",
    "Rangiku": "AgACAgUAAxkBAAIsfmmAqSgI8GwQYqg896bwyxX4dYFmAAI3DWsb0oUJVPw1vQ_dECRgAQADAgADeQADOAQ",
    "Nelliel": "AgACAgUAAxkBAAIsgGmAqS3E8km8teHNAdxrHZo2ZDQGAAI9DWsb0oUJVKR4vaUKcbJCAQADAgADeQADOAQ",
    "Rukia": "AgACAgUAAxkBAAIsgmmAqTSObQM3bjnpP7torTOJd_jDAAJFDWsb0oUJVHUD_z5xRVB8AQADAgADeAADOAQ",
    "Renji Abarai": "AgACAgUAAxkBAAIshGmAqTm6esmlOi6l-fiddwABsXE04QACRg1rG9KFCVShjZ_Ta34hZgEAAwIAA3kAAzgE",
    "Riruka": "AgACAgUAAxkBAAIshmmAqT-WPnKRkDFNf2KsC5EcoQjJAAJHDWsb0oUJVAwki4zC66QaAQADAgADeQADOAQ",
    "Yachiru": "AgACAgUAAxkBAAIsiGmAqUQi2P1sqX4FyKdmHvWjXpd_AAJJDWsb0oUJVHq3osYmTNmnAQADAgADeAADOAQ",
    "Kotetsu": "AgACAgUAAxkBAAIsimmAqUnLYWFc_yU6ySpQsfnKgV8JAAJODWsb0oUJVJ3Jafm6NIC9AQADAgADeAADOAQ",
    "Yasutora Sado": "AgACAgUAAxkBAAIsjGmAqU6vNG1QRa01eKhjgSdvORHAAAJWDWsb0oUJVNCISbFeTlAHAQADAgADeAADOAQ",
    "Shuhei hisagi": "AgACAgUAAxkBAAIsjmmAqVOdHdhUFpQeyM-Pl6zf4SO4AAJbDWsb0oUJVPhaCIrlsM1wAQADAgADeAADOAQ",
    "Ikkaku": "AgACAgUAAxkBAAIskGmAqVqv73yKeokjw-vYUisxflH3AAJcDWsb0oUJVDDrg9oe6lsuAQADAgADeAADOAQ",
    "Yumichika": "AgACAgUAAxkBAAIsk2mAqV9SK7h_XEF5U6ZAacdgSTZBAAJdDWsb0oUJVK6PUMu_Yw-SAQADAgADeAADOAQ",
    "Tetsuzaemon": "AgACAgUAAxkBAAIslWmAqWXYyg5qnlEzVags9lpfNYBAAAJnDWsb0oUJVENjHANZjQjSAQADAgADeAADOAQ",
    "Orihime Inoue": "AgACAgUAAxkBAAIsl2mAqZClDh0PEJkVeFSdqMJxm6_6AAJsDWsb0oUJVD0U1bcJxNyVAQADAgADeQADOAQ",
    "Tsukishima": "AgACAgUAAxkBAAIsmWmAqZWFsjFyvMS3cXADHoWb6EvUAAJtDWsb0oUJVNQPFqRytClUAQADAgADeAADOAQ",
    "Gremmy": "AgACAgUAAxkBAAIsm2mAqZtIYKGh4jt_vpUJeAELK5ijAAJ3DWsb0oUJVJIe_dmPcqjhAQADAgADeAADOAQ",
    "Fana": "AgACAgUAAxkBAAIsnWmAqaWxXFthfa4oMI8qZKwEyuHfAAJ4DWsb0oUJVAiBJ-gbDPLcAQADAgADeAADOAQ",
    "Vanessa": "AgACAgUAAxkBAAIsn2mAqaouXwuXBnPhCuc6qVwWHZ27AAKDDWsb0oUJVI_mhSQzIhbQAQADAgADeAADOAQ",
    "Gaja": "AgACAgUAAxkBAAIsoWmAqa9BTgGiThO85uPAxLnAabYaAAKEDWsb0oUJVNV3_MBDJ1v4AQADAgADeAADOAQ",
    "Mimosa": "AgACAgUAAxkBAAIso2mAqbMwd98TjDo8MaWh8cCvrcBUAAKHDWsb0oUJVM0eZVUpzN5ZAQADAgADeAADOAQ",
    "Zora Ideale": "AgACAgUAAxkBAAIspWmAqbiSY51AuJfgyVmTWDL4Z-FTAAKJDWsb0oUJVK0sUgTzdgILAQADAgADeAADOAQ",
    "Nero": "AgACAgUAAxkBAAIsp2mAqb5CQ4XYl_N370PObdpxt_vyAAKKDWsb0oUJVKKWgdbnPtPSAQADAgADeAADOAQ",
    "Noelle Silva": "AgACAgUAAxkBAAIsqWmAqcN8bX2GCKON-MZ5ugzuCWlKAAKLDWsb0oUJVLfXiv245rAUAQADAgADeAADOAQ",
    "Luck Voltia": "AgACAgUAAxkBAAIsq2mAqcwbEya_qaPyQMpn07qxqLBXAAKPDWsb0oUJVPgpdV8X_Yn6AQADAgADeAADOAQ",
    "Finral": "AgACAgUAAxkBAAIsrWmAqdMSWgRRYKTttpW4VucjscOBAAKRDWsb0oUJVI0x03d3hrpLAQADAgADeAADOAQ",
    "Magma": "AgACAgUAAxkBAAIsr2mAqdginB1NqalbZVvS3Hhzu3bfAAKSDWsb0oUJVHxEZK53uVlTAQADAgADeAADOAQ",
    "Langris": "AgACAgUAAxkBAAIssWmAqd1fD9imYS9QEpE58Yb9gd59AAKVDWsb0oUJVEaBK4J9mtOOAQADAgADeAADOAQ"
}

IMAGE_URLS = {
    "Yamato": "AgACAgUAAxkBAAIrbGl_XYB0TK4J67UGqAJ7K72GVRWhAAK2DGsb94QBVJk8dxXA-7hyAQADAgADeQADOAQ",
    "Eustass Kid": "AgACAgUAAxkBAAIud2mCA58ss_N8uDbjp-yOnkC6JAj8AAKBD2sbCfwQVBM3mUmj7RHHAQADAgADeQADOAQ",
    "Buggy": "AgACAgUAAxkBAAIrb2l_XZt7xmqcfFrmkBnJXtZp5j4dAAK4DGsb94QBVBinfq8obshLAQADAgADeQADOAQ",
    "Arlong": "AgACAgUAAxkBAAIrcml_XaQs5vPwGs0vezSGgvxz9s4zAAK5DGsb94QBVPxeNDcE4NrPAQADAgADeAADOAQ",
    "Koby": "AgACAgUAAxkBAAIrdWl_XasSiKHzywg5b3G7kIhHtvtoAAK6DGsb94QBVIpHVTFkNstNAQADAgADeQADOAQ",
    "Alvida": "AgACAgUAAxkBAAIreGl_XbU_P1NbZt7B84BKciNBrXRRAAK7DGsb94QBVMTiyOREkM4WAQADAgADeQADOAQ",
    "Chopper": "AgACAgUAAxkBAAIre2l_Xb0Y2RI0E44l0Nr0GXoGAh6cAAK9DGsb94QBVGen5Paut_nn2AQADAgADeQADOAQ",
    "Usopp": "AgACAgUAAxkBAAIrfml_XcfQ6mWgLwebz_Ns4jfR-XeHAAK-DGsb94QBVDGoTiUCadGIAQADAgADeQADOAQ",
    "Helmeppo": "AgACAgUAAxkBAAIrgWl_XdSazwtqkNQQ5jOoWeeJ9hrqAAK_DGsb94QBVLh6NPV1y_YZAQADAgADeQADOAQ",
    "Nami": "AgACAgUAAxkBAAIp-2l-txM84hKLMqVz6oT9z-wpc_o9AAKhDWsb94T5V_JkNM5QQs5BAQADAgADeAADOAQ",
    "Portgas D Ace": "AgACAgUAAxkBAAKvgmmQsZ5eM4bTwiNy2r--6smZN-NwAAI2E2sbEsmAVKlZrFQDhl38AQADAgADeAADOgQ",
    "Trafalgar D. Law": "AgACAgUAAxkBAAKvgGmQsYArjBEkQiUdWIG70IGHZIcnAAI1DWsbY8qAVMaAXSHYPwudAQADAgADeQADOgQ",
    "Default": "AgACAgUAAxkBAAIBXWl3kMo8CaQ8taCni8_uV3ikQiN4AAJZDWsbLpy4V86gS3f_7AWhAQADAgADeAADOAQ"
}

battles = {}
pending_explores = {}

# =====================
# LEVELING UTILS
# =====================

def get_required_char_exp(level):
    if 1 <= level <= 5: return 500
    if 6 <= level <= 10: return 1000
    if 11 <= level <= 15: return 2000
    if 16 <= level <= 20: return 2500
    if 21 <= level <= 29: return 3000
    if level >= 30 and level < 40: return 5000 # Cost to progress to 40 via exp (if allowed)
    return 9999999 # Max Level 40

def get_required_player_exp(level):
    if level >= 100: return 999999999
    if 1 <= level <= 5: return 200
    if 6 <= level <= 10: return 500
    if 11 <= level <= 20: return 1500
    if 21 <= level <= 30: return 2000
    if 31 <= level <= 70: return 3000
    if 71 <= level <= 100: return 6000
    return 10000

def check_player_levelup(p):
    lvl = p.get('level', 1)
    exp = p.get('exp', 0)
    req = get_required_player_exp(lvl)
    levels_gained = 0

    while exp >= req and lvl < 100:
        exp -= req
        lvl += 1
        levels_gained += 1
        req = get_required_player_exp(lvl)

        p['clovers'] = p.get('clovers', 0) + 10
        p['berries'] = p.get('berries', 0) + 500
        p['bounty'] = p.get('bounty', 0) + 40

    p['level'] = lvl
    p['exp'] = exp

    if p.get('level', 1) >= 10 and p.get('referred_by') and not p.get('referral_reward_claimed'):
        p['referral_reward_claimed'] = True
        child_berries, child_clovers = 2500, 25
        parent_berries, parent_clovers = 5000, 50
        p['berries'] += child_berries
        p['clovers'] += child_clovers
        
        parent_id = p.get('referred_by')
        parent = load_player(parent_id)
        if parent:
            parent['berries'] += parent_berries
            parent['clovers'] += parent_clovers
            parent['referrals'] = parent.get('referrals', 0) + 1
            save_player(parent_id, parent)
            
    return levels_gained

def check_char_levelup(char):
    lvl = char.get('level', 1)
    exp = char.get('exp', 0)
    
    # Enforce Cap at 40
    if lvl > 40:
        char['level'] = 40
        char['exp'] = 0 # Or max display, usually 0 if capped
        return

    if lvl == 40:
        char['exp'] = get_required_char_exp(40) # Show full bar
        return

    # Cap auto-leveling at 30. From 30 to 40, must use tokens.
    if lvl >= 30:
        return

    req = get_required_char_exp(lvl)
    while exp >= req and lvl < 30:
        exp -= req
        lvl += 1
        req = get_required_char_exp(lvl)
    
    char['level'] = lvl
    char['exp'] = exp

def get_scaled_stats(char_obj, player_fruit=None):
    name = char_obj['name']
    base = CHARACTERS.get(name, CHARACTERS["Usopp"])
    lvl = char_obj.get('level', 1)
    # Ensure Cap for calc
    if lvl > 40: lvl = 40
    
    bonus_multiplier = lvl - 1

    # INCREASED MULTIPLIERS to ensure stats differ significantly by level
    stats = {
        "hp": base['hp'] + (25 * bonus_multiplier),  # Increased HP scaling
        "atk_min": base['atk_min'] + (12 * bonus_multiplier), # Increased Atk scaling
        "atk_max": base['atk_max'] + (12 * bonus_multiplier),
        "def": base['def'] + (10 * bonus_multiplier),
        "spe": base['spe'] + (12 * bonus_multiplier)
    }

    if player_fruit and player_fruit in DEVIL_FRUITS:
        fruit = DEVIL_FRUITS[player_fruit]
        stats['atk_min'] += fruit['atk_buff']
        stats['atk_max'] += fruit['atk_buff']
        stats['def'] += fruit['def_buff']
        stats['hp'] += fruit['hp_buff']

    return stats

# =====================
# CORE UTILS
# =====================
def escape_md(text):
    """Escapes special characters for Telegram Markdown."""
    if not text:
        return ""
    return str(text).replace("_", "\\_").replace("*", "\\*").replace("`", "\\`").replace("[", "\\[")

async def is_spamming(user_id, cooldown_seconds=3):
    p = get_player(user_id)
    current_time = time.time()
    last_time = BUTTON_COOLDOWNS.get(user_id, 0)
    
    if current_time - last_time < cooldown_seconds:
        return True, int(cooldown_seconds - (current_time - last_time))
    
    BUTTON_COOLDOWNS[user_id] = current_time
    return False, 0

async def trigger_security_check(user_id, context):
    p = get_player(user_id)
    riddle = random.choice(RIDDLES)
    p['verification_active'] = True
    save_player(user_id, p)
    
    options = riddle['options'].copy()
    random.shuffle(options)
    
    keyboard = []
    for opt in options:
        is_correct = "1" if opt == riddle['correct'] else "0"
        keyboard.append(InlineKeyboardButton(opt, callback_data=f"v:{is_correct}:{user_id}"))

    text = (
        f"⚠️ *MARINE SECURITY CHECK!*\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Identify *{riddle['hint']}* within 30 seconds!\n"
        f"━━━━━━━━━━━━━━━━━━━"
    )
    
    try:
        msg = await context.bot.send_message(
            chat_id=user_id, 
            text=text, 
            reply_markup=InlineKeyboardMarkup([keyboard]), 
            parse_mode="Markdown"
        )
        context.job_queue.run_once(
            security_timeout, 
            30, 
            data={'user_id': user_id, 'msg_id': msg.message_id}
        )
    except Exception as e:
        p['verification_active'] = False
        save_player(user_id, p)
        logging.warning(f"Could not send security check to {user_id}: {e}")

async def security_timeout(context: ContextTypes.DEFAULT_TYPE):
    job_data = context.job.data
    uid = job_data['user_id']
    msg_id = job_data['msg_id']
    p = get_player(uid)
    
    if p and p.get('verification_active'):
        p['verification_active'] = False
        p['is_locked'] = True
        save_player(uid, p)

        try:
            await context.bot.edit_message_text(
                chat_id=uid,
                message_id=msg_id,
                text="🚫 *ACCOUNT LOCKED (TIMEOUT)*\nYou failed to respond to the Marine Security Check. Contact admin."
            )
            await context.bot.send_message(
                chat_id="-1003855697962",
                text=f"🚨 *BOT DETECTION (TIMEOUT)*\n👤: `{p.get('name')}`\n🆔: `{uid}`\n👉 `/unlock {uid}`",
                parse_mode="Markdown"
            )
        except Exception as e:
            logging.error(f"Timeout logic failed for {uid}: {e}")

async def unlock_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("🚫 Access Denied: Admins only.")
        return

    if not context.args:
        await update.message.reply_text("⚠️ Usage:\n`/unlock <id>` (Unlock one)\n`/unlock all` (Unlock EVERYONE)")
        return

    command_arg = context.args[0].lower()

    if command_arg == "all":
        await update.message.reply_text("🔄 *Unlocking ALL players...* (This may take a moment)")
        try:
            result = players_collection.update_many(
                {"is_locked": True},
                {"$set": {"is_locked": False, "verification_active": False, "last_interaction": 0}}
            )
            ram_count = 0
            for uid in player_cache:
                if player_cache[uid].get('is_locked'):
                    player_cache[uid]['is_locked'] = False
                    player_cache[uid]['verification_active'] = False
                    player_cache[uid]['last_interaction'] = 0
                    ram_count += 1
            
            msg = (
                f"✅ *GLOBAL UNLOCK COMPLETE*\n\n"
                f"📂 Database Updated: {result.modified_count} players\n"
                f"🧠 RAM Updated: {ram_count} active sessions\n"
                f"🔓 Everyone is free to sail!"
            )
            await update.message.reply_text(msg)
        except Exception as e:
            await update.message.reply_text(f"❌ Database Error: {e}")
        return

    results = []
    for target_id in context.args:
        try:
            clean_id = str(target_id).replace(",", "").strip()
            p = load_player(clean_id)
            if not p:
                results.append(f"⚠️ `{clean_id}`: Not found")
                continue
            p['is_locked'] = False
            p['verification_active'] = False
            p['last_interaction'] = 0 
            save_player(clean_id, p)
            results.append(f"✅ `{p['name']}`: Unlocked")
            try:
                await context.bot.send_message(chat_id=clean_id, text="🔓 *Account Unlocked!*\nThe Marine Security lock has been lifted.")
            except: pass 
        except Exception as e:
            results.append(f"❌ `{target_id}`: Error")

    if results:
        await update.message.reply_text("\n".join(results), parse_mode="Markdown")

def get_player(user_id, username=None):
    uid = str(user_id)
    p = load_player(uid)
    
    if not p:
        p = {
            "user_id": uid, "name": username or "Pirate", "team": [], "characters": [],
            "berries": 10000, "clovers": 0, "bounty": 100000, "exp": 0, "level": 1,
            "starter_summoned": False, "wins": 0, "losses": 0, "explore_wins": 0, "kill_count": 0,
            "fruits": [], "equipped_fruit": None, "tokens": 0, "weapons": [],
            "artifacts": [], "keys": [],
            "explore_count": 0, "start_date": datetime.now().strftime("%Y-%m-%d"),
            "referred_by": None, "referrals": 0,
            "daily_stats": {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "kills": 0, "bounty_gained": 0, "clovers_gained": 0,
                "claimed": False
            }
        }
        save_player(uid, p)
    else:
        defaults = {
            "user_id": uid, "team": [], "characters": [], "berries": 0, "clovers": 0, "bounty": 100000,
            "exp": 0, "level": 1, "wins": 0, "losses": 0, "explore_wins": 0, "kill_count": 0,
            "fruits": [], "equipped_fruit": None, "tokens": 0, "weapons": [],
            "artifacts": [], "keys": [],
            "explore_count": 0, "start_date": datetime.now().strftime("%Y-%m-%d"),
            "referred_by": None, "referrals": 0,
            "daily_stats": {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "kills": 0, "bounty_gained": 0, "clovers_gained": 0,
                "claimed": False
            }
        }
        modified = False
        for k, v in defaults.items():
            if k not in p: 
                p[k] = v
                modified = True
        
        if not p.get("name") or p["name"] == "Pirate":
            if username: 
                p["name"] = username
                modified = True
        
        if int(user_id) in ADMIN_IDS:
            p["berries"] = max(p.get("berries", 0), 99999999)
            p["clovers"] = max(p.get("clovers", 0), 99999999)
            p["level"] = 100
            modified = True

        if modified:
            save_player(uid, p)

    return p

def get_exp_bar(current, max_exp, length=12):
    if max_exp <= 0 or current >= max_exp: return "█" * length
    percent = min(1.0, current / max_exp)
    filled = int(percent * length)
    return "█" * filled + "▒" * (length - filled)

def get_stats_text(char_obj_or_name, player_fruit=None):
    if isinstance(char_obj_or_name, str):
        # Default view for generic lookup
        name = char_obj_or_name
        lvl = 1
        current_exp = 0
        equipped_weapon = None
        stats = get_scaled_stats({"name": name, "level": 1}, player_fruit)
    else:
        # User's character specific view
        name = char_obj_or_name['name']
        lvl = char_obj_or_name.get('level', 1)
        if lvl > 40: lvl = 40 # Cap for display
        
        current_exp = char_obj_or_name.get('exp', 0)
        equipped_weapon = char_obj_or_name.get('equipped_weapon')
        # Use current level for accurate stats
        stats = get_scaled_stats({"name": name, "level": lvl}, player_fruit)

    c = CHARACTERS.get(name)
    if not c: return "Character not found."

    rarity_info = RARITY_STYLES.get(c['rarity'], {"label": c['rarity']})
    rarity_display = rarity_info['label']
    
    # Moves info
    basic_move_name = c['moves'][0]
    basic_move_dmg = MOVES[basic_move_name]['dmg']
    basic_move_effect = MOVES[basic_move_name].get('effect', "None")
    
    ult_name = c['ult']
    ult_damage = MOVES[ult_name]['dmg']
    ult_desc = EFFECT_DESCRIPTIONS.get(name, "None")
    
    # REMOVED EXP BAR AND PERCENTAGE AS REQUESTED

    text = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    text += f"      ┃ {name.upper()} ┃\n"
    text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    text += f" ✦   {c['rarity']} {rarity_info.get('symbol','')}   ✦ \n"
    text += f" ✦   {c['class']}   ✦\n\n"
    
    text += f"LEVEL ▸ [ {lvl} ]\n"
    # text += f"EXP   ▸ [ {current_exp} ] / [ {req_exp} ]\n" # Removed
    # text += f"[{exp_bar}] \n" # Removed
    # text += f"[ {exp_percent}% ]\n\n" # Removed
    
    text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    text += f"          ✦ VITALS ✦\n"
    text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    text += f"  ❤️ HP  ▸ {stats['hp']}\n"
    text += f"  ⚔️ ATK ▸ {stats['atk_min']} - {stats['atk_max']}\n"
    text += f"  🛡️ DEF ▸ {stats['def']}\n"
    text += f"  ⚡ SPD ▸ {stats['spe']}\n"
    text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    text += f"        ✦ TECHNIQUES ✦\n"
    text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    text += f"➤ [ {basic_move_name} ]\n"
    text += f"   Power  : {basic_move_dmg}\n"
    text += f"   Effect : {basic_move_effect}\n\n"
    
    text += f"➤ [ {ult_name} ]\n"
    text += f"   Power  : {ult_damage}\n"
    text += f"   Effect : {ult_desc}\n\n"
    
    if equipped_weapon:
        w_data = WEAPONS.get(equipped_weapon)
        if w_data:
            text += f"➤ ⚔️ [ {equipped_weapon} ]\n"
            text += f"   Power  : {w_data['atk_val']}\n"
            text += f"   Effect : {w_data['spec']}\n"

    text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    return text


def generate_char_instance(name, level=1, player_fruit=None, equipped_weapon=None):
    # Handle aliases
    name_lower = name.lower()
    if name_lower in NAME_ALIASES:
        name = NAME_ALIASES[name_lower]

    if level > 40: level = 40

    c = CHARACTERS.get(name, {
        "hp": 300, "atk_min": 15, "atk_max": 25, "def": 15, "spe": 20,
        "moves": ["Strike", "Bash"], "ult": "Special Beam"
    })

    # Recalculate stats based on actual level
    stats = get_scaled_stats({"name": name, "level": level}, player_fruit)
    moves = list(c.get('moves', ["Strike", "Bash"]))
    
    # IMPORTANT: Append weapon move here
    if equipped_weapon and equipped_weapon in WEAPONS:
        moves.append(WEAPONS[equipped_weapon]['spec'])

    exp = 0
    if level == 40:
        exp = get_required_char_exp(40) # Max bar

    return {
        "id": str(uuid.uuid4())[:8], "name": name, "level": level, "exp": exp,
        "hp": stats['hp'], "max_hp": stats['hp'], "atk_min": stats['atk_min'],
        "atk_max": stats['atk_max'], "def": stats['def'], "spe": stats['spe'],
        "moves": moves, "ult": c.get('ult', "Special Beam"),
        "stunned": False, "ult_used": False, "dodge_chance": 0, "equipped_weapon": equipped_weapon
    }
# =====================
# EXPLORE LOGIC
# =====================

async def explore_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private":
        await update.message.reply_text("⚠️ This command can only be used in private messages (DM).")
        return

    if not load_player(update.effective_user.id):
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return

    p = get_player(update.effective_user.id)
    uid = str(p['user_id'])
    p['last_interaction'] = time.time()
    if p and p.get('is_locked'):
        await update.effective_message.reply_text("❌ Your account is locked. Contact admin.")
        return

    if uid in pending_explores:
        pending_data = pending_explores[uid]
        last_time = pending_data.get('time', 0) if isinstance(pending_data, dict) else 0

        if time.time() - last_time < 120:
            remaining = int(120 - (time.time() - last_time))
            await update.message.reply_text(f"⚠️ You have an unfinished battle! You can escape and explore again in {remaining} seconds.")
            return
        else:
            del pending_explores[uid]

    p['explore_count'] += 1
    
    # Initial RNG
    roll = random.random()

    # --- ARTIFACTS (1% Each) ---
    if roll < 0.01: # Purple Artifact
        p.setdefault("artifacts", []).append("Purple-Artifact")
        save_player(uid, p)
        await update.message.reply_photo(
            PURPLE_ARTIFACT_IMG, 
            caption="While exploring you have stumbled upon Purple-Artifact , you can see it on /inventory , also you can sell it using /sell Purple-Artifact"
        )
        return
    elif roll < 0.02: # Blue Artifact (1% chance separate)
        p.setdefault("artifacts", []).append("Blue-Artifact")
        save_player(uid, p)
        await update.message.reply_photo(
            BLUE_ARTIFACT_IMG,
            caption="You have been stumbled upon Blue-Artifact , you can see it on /inventory also you can sell it by using /sell Blue-Artifact"
        )
        return
    
    # --- KEYS (2% Chance) ---
    elif roll < 0.04: # Skull Reef-key
        p.setdefault("keys", []).append("Skull Reef-key")
        save_player(uid, p)
        await update.message.reply_text("🗝 *Key Found!*\n\nYou stumbled upon a *Skull Reef-key*! It has been added to your inventory.")
        return

    # --- CHESTS (Approx 5%) ---
    elif roll < 0.09:
        # Chests can now drop ANY weapon or fruit except Forest Blade
        # Determine contents
        chest_type = "Frost" if roll < 0.055 else ("Gold" if roll < 0.075 else "Dark")
        img = FROST_CHEST_IMG if chest_type == "Frost" else (GOLD_CHEST_IMG if chest_type == "Gold" else DARK_CHEST_IMG)
        
        c_luck = random.randint(5, 20)
        c_berry = random.randint(2000, 6000)
        c_tokens = random.randint(1, 3)
        
        p['clovers'] += c_luck
        p['berries'] += c_berry
        if c_tokens > 0: p['tokens'] += c_tokens
        
        # Add Daily Stats
        if "daily_stats" not in p: p["daily_stats"] = {}
        p["daily_stats"]["clovers_gained"] = p["daily_stats"].get("clovers_gained", 0) + c_luck
        p["daily_stats"]["bounty_gained"] = p["daily_stats"].get("bounty_gained", 0) # Berries don't count as bounty

        # Weapon/Fruit Drop Logic
        drop_roll = random.random()
        drop_text = ""
        if drop_roll < 0.4: # 40% Chance for Item Drop in chest
            is_weapon = random.choice([True, False])
            if is_weapon:
                # Exclude Forest Blade
                avail_weaps = [w for w in WEAPONS.keys() if w != "Forest Blade"]
                w_drop = random.choice(avail_weaps)
                p.setdefault("weapons", []).append(w_drop)
                drop_text = f"\n⚔️ Found Weapon: *{w_drop}*"
            else:
                f_drop = random.choice(list(DEVIL_FRUITS.keys()))
                p.setdefault("fruits", []).append(f_drop)
                drop_text = f"\n🍎 Found Fruit: *{f_drop}*"

        save_player(uid, p)
        text = f"While exploring, You found a {chest_type} Chest\n\nIt contains\n{c_luck} 🍀\n{c_berry} 🍇\n{c_tokens} Level up token🧩{drop_text}"
        await update.message.reply_photo(img, caption=text, parse_mode="Markdown")
        return

    # --- RESOURCES (Approx 15%) ---
    elif roll < 0.24:
        # Berries 1000-10000, Clovers 50-250
        is_berry = random.choice([True, False])
        if is_berry:
            amt = random.randint(1000, 10000)
            p['berries'] += amt
            text = f" You have stumbled upon {amt} berries 🍇!"
        else:
            amt = random.randint(50, 250)
            p['clovers'] += amt
            p.setdefault("daily_stats", {})["clovers_gained"] = p["daily_stats"].get("clovers_gained", 0) + amt
            text = f"🍀 You just found {amt} clovers on the ground!"
        
        save_player(uid, p)
        await update.message.reply_text(text)
        return

    # --- BATTLE (Default) ---
    wins = p.get('explore_wins', 0)
    if wins in BOSS_MISSIONS:
        boss = BOSS_MISSIONS[wins]
        char_name = boss['name']
        img_id = boss['img']
        text = (
            f"🚨 *MISSION BOSS ENCOUNTER* 🚨\n\n"
            f"You've defeated {wins} challengers! The boss *{char_name}* has appeared to block your path!\n\n"
            f"Prepare for a legendary battle!"
        )
    else:
        char_name = random.choice(list(EXPLORE_DATA.keys()))
        img_id = EXPLORE_DATA[char_name]
        text = (
            f"🧭 *EXPLORATION* 🧭\n\n"
            f"You encountered *{char_name}* while sailing the Grand Line!\n"
            f"Do you wish to engage in battle?"
        )

    pending_explores[uid] = {'name': char_name, 'time': time.time()}

    kb = [
        [InlineKeyboardButton(f"Fight {char_name} ⚔", callback_data=f"efight_{char_name}")],
        [InlineKeyboardButton("📜 Missions", callback_data="show_missions")]
    ]

    try:
        await update.message.reply_photo(
            img_id, 
            caption=text, 
            reply_markup=InlineKeyboardMarkup(kb), 
            parse_mode="Markdown"
        )
    except Exception as e:
        logging.error(f"Image failed for {char_name}: {e}")
        await update.message.reply_text(
            f"⚠️ *IMAGE ERROR* ⚠️\n(The image for {char_name} is broken, but you can still fight!)\n\n{text}",
            reply_markup=InlineKeyboardMarkup(kb),
            parse_mode="Markdown"
        )

# =====================
# STARTER, REFERRAL & NAV
# =====================

async def show_starter_page(update, name, target_user_id):
    text = get_stats_text(name)
    img = IMAGE_URLS.get(name, IMAGE_URLS["Default"])
    order = ["Usopp", "Nami", "Helmeppo"]
    if name not in order: name = "Usopp"
    idx = order.index(name)

    btns = [[InlineKeyboardButton("Choose this Pirate", callback_data=f"choose_{name}_{target_user_id}")]]
    nav = []
    if idx > 0: nav.append(InlineKeyboardButton("⬅ Previous", callback_data=f"start_{order[idx-1]}_{target_user_id}"))
    if idx < len(order) - 1: nav.append(InlineKeyboardButton("Next ➡", callback_data=f"start_{order[idx+1]}_{target_user_id}"))
    btns.append(nav)

    markup = InlineKeyboardMarkup(btns)
    try:
        if update.callback_query:
            await update.callback_query.edit_message_media(InputMediaPhoto(img, caption=text), reply_markup=markup)
        else:
            await update.message.reply_photo(img, caption=text, reply_markup=markup)
    except Exception: pass


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    username = update.effective_user.username or update.effective_user.first_name or "Pirate"

    p = load_player(user_id) 
    is_new = False

    if p and p.get('is_locked'):
        await update.effective_message.reply_text("❌ Your account is locked. Contact admin.")
        return

    if not p:
        is_new = True
        p = {
            "user_id": user_id, "name": username, "team": [], "characters": [],
            "berries": 10000, "clovers": 0, "bounty": 100000, "exp": 0, "level": 1,
            "starter_summoned": False, "wins": 0, "losses": 0, "explore_wins": 0, "kill_count": 0,
            "fruits": [], "equipped_fruit": None, "tokens": 0, "weapons": [],
            "artifacts": [], "keys": [],
            "explore_count": 0, "start_date": datetime.now().strftime("%Y-%m-%d"),"is_locked": False,
            "verification_active": False, "referred_by": None, "referrals": 0,
            "referral_reward_claimed": False,
            "daily_stats": {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "kills": 0, "bounty_gained": 0, "clovers_gained": 0,
                "claimed": False
            }
        }

    if is_new and context.args:
        try:
            referrer_id = str(context.args[0])
            if referrer_id != user_id:
                referrer = load_player(referrer_id)
                if referrer:
                    p['referred_by'] = referrer_id
                    await update.message.reply_text(
                        f"🤝 Recruited by {referrer['name']}!\n"
                        "Reach *Level 10* to unlock your starting bonus! 🎁"
                    )
        except Exception as e:
            logging.error(f"Referral logic error: {e}")

    save_player(user_id, p)

    if p.get("starter_summoned"):
        await update.message.reply_text(f"Welcome back Captain {p['name']}!")
        return

    await show_starter_page(update, "Usopp", user_id)

async def referral_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    bot_username = context.bot.username
    link = f"https://t.me/{bot_username}?start={user_id}"

    p = get_player(user_id)
    ref_count = p.get('referrals', 0)

    if p and p.get('is_locked'):
        await update.effective_message.reply_text("❌ Your account is locked. Contact admin.")
        return

    text = (
        f"╭━━━━━━━━━━━━━━━╮\n"
        f"✦    🤝 REFERRAL 🤝     ✦\n"
        f"╰━━━━━━━━━━━━━━━╯\n\n"
        f"Share your link to grow your fleet!\n"
        f"Rewards are granted when your friend reaches *Level 10*. 🏆\n\n"
        f"🔗 *YOUR LINK:*\n`{link}`\n\n"
        f"🎁 *REVISED REWARDS*\n"
        f"• You get: 5,000 🍇 + 50 🍀\n"
        f"• Friend gets: 2,500 🍇 + 25 🍀\n\n"
        f"📊 *TOTAL RECRUITS:* `{ref_count}`"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


# =====================
# STORE & BUY SYSTEM
# =====================

async def store_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private":
        await update.message.reply_text("⚠️ This command can only be used in private messages (DM).")
        return
    uid = str(update.effective_user.id)
    p = load_player(uid)
    
    if p and p.get('is_locked'):
        await update.effective_message.reply_text("❌ Your account is locked. Contact admin.")
        return

    if not load_player(update.effective_user.id):
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return

    text = "⚓️ *PIRATE STORE* ⚓️\n\nWelcome to the black market. Select a category to browse items."
    kb = [
        [InlineKeyboardButton("Weapons ⚔️", callback_data="store_weapons"), InlineKeyboardButton("Fruits 🍎", callback_data="store_fruits")],
        [InlineKeyboardButton("Close", callback_data="wheel_cancel")]
    ]
    await update.message.reply_photo(STORE_IMG, caption=text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

async def handle_store_callback(query, category):
    if category == "weapons":
        text = "⚔️ *WEAPONS FOR SALE* ⚔️\n\n"
        for name, d in WEAPONS.items():
            text += f"• *{name}*: 🍇{d['cost']:,} (Rank {d['lvl']}+)\n"
        text += "\nUse `/buy Item Name` to purchase."
    else:
        text = "🍎 **DEVIL FRUITS FOR SALE** 🍎\n\n"
        for name, d in DEVIL_FRUITS.items():
            text += f"• *{name}*: 🍇{d['cost']:,} (Rank {d['lvl']}+)\n"
        text += "\nUse `/buy Item Name` to purchase."

    # FIX: Removed trailing comma that was causing tuple error
    kb = [[InlineKeyboardButton("Back to Store", callback_data="back_to_store")]]
    await query.edit_message_caption(caption=text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/buy Item Name`")
        return
    uid = str(update.effective_user.id)
    p = load_player(uid)
    
    if p and p.get('is_locked'):
        await update.effective_message.reply_text("❌ Your account is locked. Contact admin.")
        return

    if not load_player(update.effective_user.id):
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return

    input_item = " ".join(context.args).lower().strip()
    p = get_player(update.effective_user.id)

    target_name = None
    item_type = None

    for w_name in WEAPONS:
        if w_name.lower() == input_item:
            target_name = w_name
            item_type = "weapon"
            break
    if not target_name:
        for f_name in DEVIL_FRUITS:
            if f_name.lower() == input_item:
                target_name = f_name
                item_type = "fruit"
                break

    if not target_name:
        await update.message.reply_text("Item not found in store.")
        return

    item_data = WEAPONS[target_name] if item_type == "weapon" else DEVIL_FRUITS[target_name]
    req_lvl = item_data['lvl']

    if p.get('level', 1) < req_lvl:
        await update.message.reply_text(f"❌ You need Player Rank {req_lvl} to purchase {target_name}!")
        return

    if item_type == "weapon":
        w = WEAPONS[target_name]
        text = (f"➥Name: {target_name}\n➥Rarity: {w['rarity']}\n➥Attack: {w['atk_range']}\n"
                f"➥Critical chance: {w['crit']}\n➥Accuracy: {w['acc']}\n"
                f"➥Special attack: {w['spec']}\n➥Rank requirement: {w['lvl']}\n\n➥ Cost: {w['cost']}🍇")
        img = w['img']
    else:
        f = DEVIL_FRUITS[target_name]
        text = f['text'] + f"\n\n➥ Cost: {f['cost']}🍇"
        img = f['img']

    kb = [[InlineKeyboardButton("Confirm Purchase ✅", callback_data=f"confbuy|{item_type}|{target_name}")]]
    await update.message.reply_photo(img, caption=text, reply_markup=InlineKeyboardMarkup(kb))

async def use_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/use Item Name`")
        return
    uid = str(update.effective_user.id)
    p = load_player(uid)
    
    if p and p.get('is_locked'):
        await update.effective_message.reply_text("❌ Your account is locked. Contact admin.")
        return


    if not load_player(update.effective_user.id):
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return

    input_name = " ".join(context.args).lower().strip()
    p = get_player(update.effective_user.id)

    if "level-up token" in input_name or "level up token" in input_name:
        if p.get('tokens', 0) <= 0:
            await update.message.reply_text("You don't have any Level-up tokens!")
            return
        kb = []
        for i, char in enumerate(p.get('characters', [])):
            if char.get('level', 1) < 40: # Max level 40
                kb.append([InlineKeyboardButton(f"{char['name']} (Lv.{char['level']})", callback_data=f"usetoken|{i}")])
        if not kb:
            await update.message.reply_text("You have no pirates to level up (Max Level 40 reached for all).")
            return
        await update.message.reply_text("Select a pirate to level up using 1 token (Max Lvl 40):", reply_markup=InlineKeyboardMarkup(kb))
        return

    target_fruit = None
    for f_name in p.get('fruits', []):
        if f_name.lower() == input_name:
            target_fruit = f_name
            break

    if target_fruit:
        p['fruits'].remove(target_fruit)
        p['equipped_fruit'] = target_fruit
        save_player(p['user_id'], p)
        await update.message.reply_text(f"✨ {target_fruit} consumed! This devil fruit's abilities have been added to your whole team.")
        return

    target_weapon = None
    for w_name in p.get('weapons', []):
        if w_name.lower() == input_name:
            target_weapon = w_name
            break

    if target_weapon:
        kb = []
        for i, char in enumerate(p.get('characters', [])):
            eq = " (Equipped)" if char.get('equipped_weapon') == target_weapon else ""
            kb.append([InlineKeyboardButton(f"{char['name']} (Lv.{char['level']}){eq}", callback_data=f"wepattach|{target_weapon}|{i}")])
        await update.message.reply_text(f"Select a character to equip *{target_weapon}* (Weapon will be consumed):", reply_markup=InlineKeyboardMarkup(kb))
        return

    await update.message.reply_text("You don't own this item or it's not usable.")

async def sell_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/sell Item Name`\n(Currently only Artifacts can be sold)")
        return
    
    uid = str(update.effective_user.id)
    p = load_player(uid)
    if not p:
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return
    
    if p.get('is_locked'):
        await update.message.reply_text("❌ Account Locked.")
        return

    item_name = " ".join(context.args).title()
    if item_name not in SELL_PRICES:
        await update.message.reply_text("⚠️ This item cannot be sold or doesn't exist.")
        return
    
    if item_name not in p.get('artifacts', []):
        await update.message.reply_text(f"⚠️ You don't have any {item_name}.")
        return

    price = SELL_PRICES[item_name]
    p['artifacts'].remove(item_name)
    p['berries'] += price
    save_player(uid, p)
    
    await update.message.reply_text(f"💰 Sold *{item_name}* for {price:,} Berries!")

# =====================
# TEAM MANAGEMENT
# =====================

async def myteam(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private":
        await update.message.reply_text("⚠️ This command can only be used in private messages (DM).")
        return
    uid = str(update.effective_user.id)
    p = load_player(uid)
    if p and p.get('is_locked'):
        await update.effective_message.reply_text("❌ Your account is locked. Contact admin.")
        return


    if not load_player(update.effective_user.id):
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return

    p = get_player(update.effective_user.id)
    team_names = ", ".join([c['name'] for c in p.get('team', [])]) or "None"
    txt = f"⚓️ YOUR TEAM ⚓️\n\nActive: {team_names}\n\nSelect up to 3 pirates for battle."
    kb = [[InlineKeyboardButton("Set Team ⚔", callback_data="manage_team")]]
    await update.message.reply_text(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

async def manage_team(query, p):
    chars = p.get("characters", [])
    
    if not chars:
        await query.answer("You have no pirates! Use /wheel first.", show_alert=True)
        return
    kb = []
    for i, c in enumerate(chars):
        status = "✅" if any(tc['id'] == c['id'] for tc in p.get('team', [])) else "❌"
        kb.append([InlineKeyboardButton(f"{c['name']} (Lv.{c['level']}) {status}", callback_data=f"toggle_{i}")])
    kb.append([InlineKeyboardButton("💾 Save Team", callback_data="save_team")])

    text = "Select up to 3 characters:"
    try:
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb))
    except:
        try:
            await query.edit_message_caption(caption=text, reply_markup=InlineKeyboardMarkup(kb))
        except:
            await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb))

# =====================
# BATTLE LOGIC
# =====================

async def battle_timeout_check(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    bid = job.data['bid']
    if bid in battles:
        b = battles[bid]
        if b['last_move_time'] == job.data['last_time']:
            quitter_p = b['turn_owner']
            winner_p = "p2" if quitter_p == "p1" else "p1"
            winner_name = b[f'{winner_p}_name']
            quitter_name = b[f'{quitter_p}_name']

            try:
                await context.bot.edit_message_text(
                    chat_id=job.chat_id,
                    message_id=job.data['msg_id'],
                    text=f"⏰ *TIMEOUT!*\n\n*{quitter_name}* took too long to move! *{winner_name}* wins by default!",
                    parse_mode="Markdown"
                )
            except: pass
            if bid in battles: del battles[bid]

async def battle_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await update.message.reply_text("Reply to someone to challenge them!")
        return

    if not load_player(update.effective_user.id):
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return

    p1_id = str(update.effective_user.id)
    p2_id = str(update.message.reply_to_message.from_user.id)
    if p1_id == p2_id: return

    if not load_player(p2_id):
        await update.message.reply_text("⚠️ Your opponent hasn't started their journey yet!")
        return

    for b in battles.values():
        if p1_id in [str(b['p1_id']), str(b.get('p2_id'))] or p2_id in [str(b['p1_id']), str(b.get('p2_id'))]:
            await update.message.reply_text("One of the players is already in a battle!")
            return

    p1, p2 = get_player(p1_id), get_player(p2_id)
    if not p1.get('team') or not p2.get('team'):
        await update.message.reply_text("Both players must have a team set via /myteam!")
        return
    kb = [[InlineKeyboardButton("Accept Battle ⚔", callback_data=f"accept_{p1_id}_{p2_id}")]]
    await update.message.reply_text(f"Hey {escape_md(p2['name'])}, {escape_md(p1['name'])} challenged you!", reply_markup=InlineKeyboardMarkup(kb))

def get_bar(h, m):
    if m <= 0: return "▒" * 10
    ratio = max(0, min(1, h/m))
    filled = int(ratio * 10)
    return "█" * filled + "▒" * (10 - filled)

async def run_battle_turn(query, battle_id, move_name=None, context=None):
    b = battles.get(battle_id)
    if not b: return

    b['last_move_time'] = time.time()
    p1_char = b['p1_team'][b['p1_idx']]
    p2_char = b['p2_team'][b['p2_idx']]

    if b['turn_owner'] == "p1":
        attacker, defender, att_p, def_p, att_team = p1_char, p2_char, "p1", "p2", b['p1_team']
    else:
        attacker, defender, att_p, def_p, att_team = p2_char, p1_char, "p2", "p1", b['p2_team']

    if attacker.get('stunned'):
        attacker['stunned'] = False
        log = f"💫 **{attacker['name']}** is stunned and skipped their turn!"
        b['turn_owner'] = def_p
        await show_move_selection(query, battle_id, log, context)
        if b.get('is_npc') and b['turn_owner'] == "p2":
            await asyncio.sleep(0.5) 
            await run_battle_turn(query, battle_id, move_name=None, context=context)
        return

    if b.get('is_npc') and b['turn_owner'] == "p2":
        basic_move = attacker['moves'][0] 
        if not attacker.get('ult_used') and random.random() < 0.3:
            move_name = attacker['ult']
        else:
            move_name = basic_move

    if not move_name:
        await show_move_selection(query, battle_id, context=context)
        return

    if random.random() < (attacker.get('dodge_chance', 0) / 100):
        log = f"💨 *{defender['name']}* dodged the attack!"
        attacker['dodge_chance'] = 0
    else:
        move_data = MOVES.get(move_name, MOVES["Strike"])
        is_ult = (move_name == attacker['ult'])
        
        if is_ult:
            attacker['ult_used'] = True
            if attacker['name'] == "Yamato":
                img = YAMATO_EXPLORE_ULT if b.get('is_npc') else YAMATO_ULT_VIDEO
                try: await query.message.reply_photo(photo=img, caption="⚡️ *THUNDER BAGUA!*")
                except: pass
            elif attacker['name'] == "Eustass Kid":
                img = KID_EXPLORE_ULT if b.get('is_npc') else KID_ULT_VIDEO
                try: await query.message.reply_photo(photo=img, caption="⚡️ *DAMNED PUNK!*")
                except: pass

        # === UPDATED DAMAGE FORMULA & CLAMPING ===
        atk_stat = random.randint(attacker.get('atk_min', 20), attacker.get('atk_max', 30))
        def_stat = defender.get('def', 10)
        move_power = move_data.get('dmg', 30)
        
        level_factor = 1 + (attacker.get('level', 1) / 10.0)
        
        base_dmg = (atk_stat * 1.5) - (def_stat * 0.8) + move_power
        
        # New Equation Constraints: Min 50, Max 850
        damage = int(max(50, min(850, base_dmg * level_factor)))
        
        defender['hp'] -= damage
        log = f"🔥 *{attacker['name']}* uses *{move_name}*!\n💥 Deals *{damage}* DMG!"

        effect = move_data.get('effect')
        if effect:
            if effect == "def_buff_10": attacker['def'] += 10
            elif effect == "team_heal_50":
                for char in att_team: char['hp'] = min(char['max_hp'], char['hp'] + 50)
            elif effect == "dodge_30": attacker['dodge_chance'] = 30
            elif effect == "stun_1": defender['stunned'] = True
            elif effect == "ace_ult":
                attacker['spe'] += int(attacker['spe'] * 0.4)
                defender['def'] = int(defender['def'] * 0.7)
            elif effect == "law_ult":
                attacker['spe'] += int(attacker['spe'] * 0.4)
                attacker['def'] += int(attacker['def'] * 0.3)
                defender['atk_min'] = int(defender['atk_min'] * 0.85)
                defender['atk_max'] = int(defender['atk_max'] * 0.85)

    if defender['hp'] <= 0:
        defender['hp'] = 0
        b[f'{def_p}_idx'] += 1
        log += f"\n\n💀 *{defender['name']}* HAS FALLEN!"

        if b[f'{def_p}_idx'] >= len(b[f'{def_p}_team']):
            winner_name = escape_md(b['p1_name'] if def_p == "p2" else b['p2_name'])
            loser_name = escape_md(b['p2_name'] if def_p == "p2" else b['p1_name'])
            rank_up_section = ""

            if b.get('is_npc'):
                uid = str(b['p1_id'])
                if uid in pending_explores: del pending_explores[uid]
                p = get_player(uid)
                
                wins_at = p.get('explore_wins', 0)
                if wins_at in BOSS_MISSIONS:
                    exp_gain, berry_gain, clover_gain, bounty_gain = random.randint(200,300), random.randint(200,250), random.randint(5,10), random.randint(100,200)
                else:
                    exp_gain, berry_gain, clover_gain, bounty_gain = random.randint(50,100), random.randint(50,100), random.randint(1,3), random.randint(20,30)
                
                p['explore_wins'] += 1
                p['kill_count'] = p.get('kill_count', 0) + 1
                p['exp'] += exp_gain; p['berries'] += berry_gain; p['clovers'] += clover_gain; p['bounty'] += bounty_gain
                
                # Daily Stats
                if "daily_stats" not in p: p["daily_stats"] = {}
                p["daily_stats"]["kills"] = p["daily_stats"].get("kills", 0) + 1
                p["daily_stats"]["bounty_gained"] = p["daily_stats"].get("bounty_gained", 0) + bounty_gain
                p["daily_stats"]["clovers_gained"] = p["daily_stats"].get("clovers_gained", 0) + clover_gain

                for team_char in b['p1_team']:
                    for main_char in p.get('characters', []):
                        if main_char['name'] == team_char['name']:
                            # Handle Max Level Logic
                            if main_char.get('level', 1) < 40:
                                main_char['exp'] = main_char.get('exp', 0) + exp_gain
                                check_char_levelup(main_char)

                lvls = check_player_levelup(p)
                if lvls > 0: 
                    rank_up_section = (
                        f"\n\n🎊 *RANK UP!* You reached *Level {p['level']}*!\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"🎁 *LEVEL UP REWARDS*:\n"
                        f"🍇 Berries: `+{lvls * 500}`\n"
                        f"🍀 Clovers: `+{lvls * 10}`\n"
                        f"฿ Bounty: `+{lvls * 40}`"
                    )
                
                save_player(uid, p)

                final_ui = (
                    f"◈☰☰☰⚔️ ＢＡＴＴＬＥ ＲＥＳＵＬＴ ⚔️☰☰☰◈\n\n"
                    f"🏆 *{winner_name}* defeated *{loser_name}*!\n\n"
                    f"📦 *LOOT DROPPED*:\n"
                    f"🌟 EXP: `+{exp_gain}`\n"
                    f"🍇 Berries: `+{berry_gain}`\n"
                    f"🍀 Clovers: `+{clover_gain}`\n"
                    f"฿ Bounty: `+{bounty_gain}`"
                    f"{rank_up_section}"
                )
            else:
                # PvP Logic
                wp_id = b['p1_id'] if def_p == "p2" else b['p2_id']
                lp_id = b['p2_id'] if def_p == "p2" else b['p1_id']
                
                wp = get_player(wp_id); wp['wins'] += 1
                lp = get_player(lp_id); lp['losses'] += 1
                
                # Bounty Logic (Only if > Level 5)
                bounty_msg = ""
                if wp.get('level', 1) > 5 and lp.get('level', 1) > 5:
                    wp['bounty'] += 10000
                    lp['bounty'] = max(0, lp.get('bounty', 0) - 10000)
                    
                    # Add to Daily Stats for Winner
                    if "daily_stats" not in wp: wp["daily_stats"] = {}
                    wp["daily_stats"]["bounty_gained"] = wp["daily_stats"].get("bounty_gained", 0) + 10000

                    bounty_msg = "\n\n💰 *BOUNTY UPDATE*:\nWinner: +10,000฿\nLoser: -10,000฿"

                save_player(wp_id, wp)
                save_player(lp_id, lp)
                
                final_ui = f"🏆 *{winner_name}* triumphed in PvP!{bounty_msg}"

            if battle_id in battles: del battles[battle_id]

            try:
                await query.edit_message_caption(caption=final_ui, parse_mode="Markdown")
            except Exception:
                try: await query.edit_message_text(final_ui, parse_mode="Markdown")
                except: await query.message.reply_text(final_ui)
            return
            
    b['turn_owner'] = def_p
    await show_move_selection(query, battle_id, log, context)
    
    if b.get('is_npc') and b['turn_owner'] == "p2":
        await asyncio.sleep(0.5) 
        await run_battle_turn(query, battle_id, move_name=None, context=context)


async def show_move_selection(query, battle_id, log="", context=None):
    b = battles.get(battle_id)
    if not b: return
    p1_char = b['p1_team'][b['p1_idx']]; p2_char = b['p2_team'][b['p2_idx']]
    attacker = b[b['turn_owner'] + '_team'][b[b['turn_owner'] + '_idx']]

    ult_name = attacker['ult']
    ult_desc = EFFECT_DESCRIPTIONS.get(attacker['name'], "Standard massive damage.")

    status = (
        f"⚔️ *ARENA* ⚔️\n━━━━━━━━━━━━━━━━━━\n"
        f"👤 *{escape_md(b['p1_name']).upper()} - {p1_char['name']}*: {p1_char['hp']}/{p1_char['max_hp']}\n"
        f"`{get_bar(p1_char['hp'], p1_char['max_hp'])}`\n\n"
        f"👤 *{escape_md(b['p2_name']).upper()} - {p2_char['name']}*: {p2_char['hp']}/{p2_char['max_hp']}\n"
        f"`{get_bar(p2_char['hp'], p2_char['max_hp'])}`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"⚡️ *{attacker['name'].upper()}'S TURN* ⚡️\n"
    )
    
    status += (
        f"🌟 *ULTIMATE*: {ult_name}\n"
        f"└─ *{ult_desc}*\n"
        f"━━━━━━━━━━━━━━━━━━\n{log if log else 'Waiting for your move...'}\n"
        f"━━━━━━━━━━━━━━━━━━\n⌛️ TURN: *{escape_md(b[b['turn_owner'] + '_name'])}*"
    )

    # === DYNAMIC MOVE BUTTONS ===
    kb = []
    moves_row = []
    # Create buttons for all moves (Basic, Moves from leveling, Weapon Moves)
    for move in attacker['moves']:
        moves_row.append(InlineKeyboardButton(f"⚔️ {move}", callback_data=f"bmove|{battle_id}|{move}"))
        if len(moves_row) == 2:
            kb.append(moves_row)
            moves_row = []
    if moves_row:
        kb.append(moves_row)

    kb.append([InlineKeyboardButton(f"🌟 ULTIMATE: {ult_name} 🌟" if not attacker.get('ult_used') else "🚫 ULTIMATE DEPLETED", 
                                    callback_data=f"bmove|{battle_id}|{ult_name}" if not attacker.get('ult_used') else "none")])
    
    kb.append([InlineKeyboardButton("🏃 Run", callback_data=f"brun_{battle_id}"), 
               InlineKeyboardButton("🏳 Forfeit", callback_data=f"bforfeit_{battle_id}")])

    try:
        msg = await query.edit_message_text(status, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    except:
        try: msg = await query.edit_message_caption(caption=status, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
        except: return

    if context and hasattr(context, 'job_queue') and context.job_queue and not b.get('is_npc'):
        context.job_queue.run_once(battle_timeout_check, 120, data={'bid': battle_id, 'last_time': b['last_move_time'], 'msg_id': msg.message_id}, chat_id=query.message.chat_id)

# =====================
# CALLBACK MASTER
# =====================

async def main_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    uid = str(query.from_user.id)
    data = query.data
    
    p = load_player(uid)
    if not p and not data.startswith("choose_") and not data.startswith("v:"):
        await query.answer("⚠️ Start your journey first! Use /start.", show_alert=True)
        return

    spamming, wait_time = await is_spamming(uid, 2)
    if spamming:
        await query.answer(f"⏳ Slow down! Wait {wait_time}s...", show_alert=False)
        return
    if p: p['last_interaction'] = time.time()

    if p and p.get('is_locked') and not data.startswith("v:"):
        await query.answer("🚫 Account Locked! Contact Admin.", show_alert=True)
        return

    if data.startswith("v:"):
        _, is_correct, target_uid = data.split(":")
        
        if uid != target_uid:
            await query.answer("❌ This check isn't for you!", show_alert=True)
            return

        if not p or not p.get('verification_active'): 
            await query.answer("⌛ This check has expired.")
            await query.message.delete()
            return

        if is_correct == "1":
            p['verification_active'] = False
            save_player(uid, p)
            await query.edit_message_text("✅ *Verification Passed!*\nContinue your journey.")
        else:
            p['is_locked'] = True
            p['verification_active'] = False
            save_player(uid, p)
            await query.edit_message_text("🚫 *ACCOUNT LOCKED.*\nContact owner to prove your identity.")
            
            await context.bot.send_message(
                chat_id="-1003855697962",
                text=f"🚨 *BOT ALERT*\n👤: `{p.get('name')}`\n🆔: `{uid}`\n❌: Failed Emoji\n👉 `/unlock {uid}`",
                parse_mode="Markdown"
            )
        return

    if data == "none":
        await query.answer("Ultimate can only be used once!")
        return

    if data == "go_shop":
        await query.answer("Visit the /store to purchase this item!", show_alert=True)
        return

    if data.startswith("start_"):
        parts = data.split("_")
        if len(parts) > 2:
            target_id = parts[2]
            if str(uid) != str(target_id):
                await query.answer("This menu is not for you!", show_alert=True)
                return
            await show_starter_page(update, parts[1], target_id)

    elif data.startswith("choose_"):
        parts = data.split("_")
        if len(parts) > 2:
            target_id = parts[2]
            if str(uid) != str(target_id):
                await query.answer("This menu is not for you!", show_alert=True)
                return

        if p.get("starter_summoned"): return
        name = parts[1]
        p.setdefault("characters", []).append(generate_char_instance(name))
        p["starter_summoned"] = True
        save_player(uid, p)
        await query.message.edit_caption(caption=f"✅ You chose *{name}*!")
    elif data == "manage_team": await manage_team(query, p)
    elif data.startswith("toggle_"):
        idx = int(data.split("_")[1])
        if idx < len(p["characters"]):
            char = p["characters"][idx]
            if any(tc['id'] == char['id'] for tc in p.get('team', [])):
                p['team'] = [tc for tc in p.get('team', []) if tc['id'] != char['id']]
            elif len(p.get('team', [])) < 3:
                if "team" not in p: p["team"] = []
                p['team'].append(char)
            save_player(uid, p); await manage_team(query, p)
    elif data == "save_team":
        await query.message.delete()
        await query.message.chat.send_message(f"Team saved! ({len(p.get('team', []))} pirates)")
    elif data == "show_missions":
        wins = p.get('explore_wins', 0)
        upcoming = [w for w in BOSS_MISSIONS.keys() if w > wins]
        m_text = f"📜 *MISSION*\n\nTarget: Defeat {min(upcoming) if upcoming else 'Max'} enemies.\nProgress: {wins}"
        await query.answer(m_text, show_alert=True)
    elif data.startswith("efight_"):
        npc_name = data.split("_", 1)[1]
        if not p.get('team'):
            await query.answer("Set your team first using /myteam!", show_alert=True); return
        bid = f"explore_{uid}"
        # Ensure fresh instances with correct stats are generated for battle
        battles[bid] = {
            "p1_id": uid, "p2_id": "NPC",
            "p1_team": [generate_char_instance(c['name'], c.get('level', 1), p.get('equipped_fruit'), c.get('equipped_weapon')) for c in p['team']],
            "p2_team": [generate_char_instance(npc_name)], "p1_idx": 0, "p2_idx": 0,
            "p1_name": p['name'], "p2_name": npc_name, "turn_owner": "p1", "is_npc": True,
            "run_votes": set(), "last_move_time": time.time()
        }
        await run_battle_turn(query, bid, move_name=None, context=context)
    elif data.startswith("accept_"):
        parts = data.split("_"); p1_id, p2_id = parts[1], parts[2]
        if uid != p2_id:
            await query.answer("This challenge isn't for you!", show_alert=True); return
        p1, p2 = get_player(p1_id), get_player(p2_id); bid = f"{p1_id}_{p2_id}"
        starter = "p1" if p1['team'][0].get('spe', 0) >= p2['team'][0].get('spe', 0) else "p2"
        battles[bid] = {
            "p1_id": p1_id, "p2_id": p2_id,
            "p1_team": [generate_char_instance(c['name'], c.get('level', 1), p1.get('equipped_fruit'), c.get('equipped_weapon')) for c in p1['team']],
            "p2_team": [generate_char_instance(c['name'], c.get('level', 1), p2.get('equipped_fruit'), c.get('equipped_weapon')) for c in p2['team']],
            "p1_idx": 0, "p2_idx": 0, "p1_name": p1['name'], "p2_name": p2['name'],
            "turn_owner": starter, "run_votes": set(), "last_move_time": time.time()
        }
        await run_battle_turn(query, bid, move_name=None, context=context)
    elif data.startswith("bmove|"):
        try: await query.answer()
        except: pass
        parts = data.split("|"); bid, move_name = parts[1], parts[2]; b = battles.get(bid)
        if not b: return
        current_turn_id = str(b['p1_id']) if b['turn_owner'] == "p1" else str(b.get('p2_id', "NPC"))
        if uid != current_turn_id and current_turn_id != "NPC":
            await query.answer("It's not your turn!", show_alert=True); return
        await run_battle_turn(query, bid, move_name, context=context)
    elif data.startswith("brun_"):
        bid = data.replace("brun_", ""); b = battles.get(bid)
        if not b or uid not in [str(b['p1_id']), str(b.get('p2_id'))]: return
        if b.get('is_npc'):
            try: await query.edit_message_caption(caption="🤝 You escaped safely.");
            except: await query.edit_message_text("🤝 You escaped safely.")
            if bid in battles: del battles[bid]
            if uid in pending_explores: del pending_explores[uid]
            return
        b['run_votes'].add(uid)
        if len(b['run_votes']) >= 2:
            try: await query.edit_message_text("🤝 Both players decided to stop.");
            except: await query.edit_message_caption(caption="🤝 Both players decided to stop.");
            if bid in battles: del battles[bid]
        else: await query.answer("Waiting for the other player...", show_alert=True)
    elif data.startswith("bforfeit_"):
        bid = data.replace("bforfeit_", ""); b = battles.get(bid)
        if not b or uid not in [str(b['p1_id']), str(b.get('p2_id'))]: return
        name = b['p1_name'] if uid == str(b['p1_id']) else b['p2_name']
        try: await query.edit_message_text(f"🏳 {name} ran away!");
        except: await query.edit_message_caption(caption=f"🏳 {name} ran away!");
        if bid in battles: del battles[bid]
        if b.get('is_npc') and uid in pending_explores: del pending_explores[uid]
    elif data == "store_weapons": await handle_store_callback(query, "weapons")
    elif data == "store_fruits": await handle_store_callback(query, "fruits")
    elif data == "back_to_store":
        kb = [[InlineKeyboardButton("Weapons ⚔️", callback_data="store_weapons"), InlineKeyboardButton("Fruits 🍎", callback_data="store_fruits")], [InlineKeyboardButton("Close", callback_data="wheel_cancel")]]
        await query.edit_message_caption(caption="⚓️ *PIRATE STORE* ⚓️\n\nWelcome back. Choose a category.", reply_markup=InlineKeyboardMarkup(kb))
    elif data.startswith("confbuy|"):
        _, itype, iname = data.split("|")
        if itype == "weapon":
            cost = WEAPONS[iname]['cost']
            if p['berries'] >= cost:
                p['berries'] -= cost
                p.setdefault('weapons', []).append(iname)
                await query.answer(f"Bought {iname}!", show_alert=True)
            else: await query.answer("Not enough berries!", show_alert=True)
        else:
            cost = DEVIL_FRUITS[iname]['cost']
            if p['berries'] >= cost:
                p['berries'] -= cost
                p.setdefault('fruits', []).append(iname)
                await query.answer(f"Bought {iname}!", show_alert=True)
            else: await query.answer("Not enough berries!", show_alert=True)
        save_player(uid, p)
    elif data.startswith("inv_"):
        # Security Check
        parts = data.split("|")
        section = parts[0].replace("inv_", "")
        
        # Check if owner_id exists in callback data (for new buttons)
        if len(parts) > 1:
            owner_id = parts[1]
            if str(uid) != str(owner_id):
                await query.answer("Unauthorized used only", show_alert=True)
                return
        
        # Logic
        kb_back = [[InlineKeyboardButton("Back", callback_data=f"back_inv|{uid}")]]
        
        if section == "weapons":
            txt = "⚔️ *YOUR WEAPONS* ⚔️\n\n"
            for w in p.get('weapons', []): txt += f"• {w}\n"
            await query.edit_message_caption(caption=txt or "No weapons.", reply_markup=InlineKeyboardMarkup(kb_back))
            
        elif section == "fruits":
            txt = "🍎 *YOUR DEVIL FRUITS* 🍎\n\n"
            for f in p.get('fruits', []): txt += f"• {f}\n"
            await query.edit_message_caption(caption=txt or "No fruits.", reply_markup=InlineKeyboardMarkup(kb_back))
            
        elif section == "artifacts":
            txt = "🏆 *YOUR ARTIFACTS* 🏆\n\n"
            for a in p.get('artifacts', []): txt += f"• {a}\n"
            if not p.get('artifacts'): txt = "No artifacts found."
            await query.edit_message_caption(caption=txt, reply_markup=InlineKeyboardMarkup(kb_back))
            
        elif section == "keys":
            txt = "🗝 *YOUR KEYS* 🗝\n\n"
            for k in p.get('keys', []): txt += f"• {k}\n"
            if not p.get('keys'): txt = "No keys found."
            await query.edit_message_caption(caption=txt, reply_markup=InlineKeyboardMarkup(kb_back))

    elif data.startswith("back_inv"):
        parts = data.split("|")
        if len(parts) > 1 and str(uid) != str(parts[1]):
            await query.answer("Unauthorized used only", show_alert=True)
            return
        await inventory_cmd(update, context, is_cb=True)

    elif data.startswith("wepattach|"):
        parts = data.split("|"); w_name, c_idx = parts[1], int(parts[2])
        if w_name in p.get('weapons', []):
            p['weapons'].remove(w_name)
            p['characters'][c_idx]['equipped_weapon'] = w_name
            save_player(uid, p); await query.edit_message_text(f"✅ Character *{p['characters'][c_idx]['name']}* now wields **{w_name}**! (Weapon consumed from inventory)")
        else:
            await query.answer("You don't own this weapon anymore!", show_alert=True)
    elif data.startswith("usetoken|"):
        c_idx = int(data.split("|")[1])
        if p.get('tokens', 0) > 0:
            char = p['characters'][c_idx]
            if char.get('level', 1) >= 40:
                await query.answer("Character is already Max Level (40)!", show_alert=True)
                return
                
            p['tokens'] -= 1
            char['level'] = char.get('level', 1) + 1
            save_player(uid, p)
            await query.edit_message_text(f"✨ Success! *{char['name']}* has reached Level {char['level']}!")
        else:
            await query.answer("No tokens left!", show_alert=True)
    elif data == "char_wheel": await wheel_options(query, "Character")
    elif data == "res_wheel": await wheel_options(query, "Resource")
    elif data.startswith("wheel_1"):
        await handle_wheel(query, p, 1, data.split("_")[2])
    elif data.startswith("wheel_5"):
        await handle_wheel(query, p, 5, data.split("_")[2])
    elif data == "wheel_cancel": await query.message.delete()
    elif data == "wheel_prob":
        await query.answer("Unknown", show_alert=True) # Updated as requested
    elif data.startswith("equip_"):
        fname = data.split("_", 1)[1]
        if fname in p.get('fruits', []):
            p['fruits'].remove(fname)
            p['equipped_fruit'] = fname
            save_player(uid, p)
            await query.answer(f"Equipped {fname}! (Consumed)", show_alert=True)
            await query.message.edit_caption(caption=f"✨ Entire team buffed by {fname}!")
        else:
            await query.answer("Fruit no longer in inventory!", show_alert=True)
    elif data == "claim_daily":
        today = datetime.now().strftime("%Y-%m-%d")
        stats = p.get("daily_stats", {})
        
        if stats.get("date") != today:
            await query.answer("Daily stats reset. Do tasks again.", show_alert=True)
            return

        if stats.get("claimed"):
            await query.answer("Already claimed today!", show_alert=True)
            return

        # Double check requirements
        c1 = stats.get("kills", 0) >= 50
        c2 = stats.get("bounty_gained", 0) >= 100000
        c3 = stats.get("clovers_gained", 0) >= 100
        
        if c1 and c2 and c3:
            p["daily_stats"]["claimed"] = True
            p["clovers"] += 300
            save_player(uid, p)
            await query.edit_message_text("✅ *DAILY REWARD CLAIMED*\n\nReceived: 300 🍀")
        else:
            await query.answer("Tasks incomplete!", show_alert=True)

# =====================
# WHEEL LOGIC
# =====================

async def wheel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not load_player(update.effective_user.id):
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return
    uid = str(update.effective_user.id)
    p = load_player(uid)
    if p and p.get('is_locked'):
        await update.effective_message.reply_text("❌ Your account is locked. Contact admin.")
        return


    desc = "🎡 **PIRATE WHEELS** 🎡\n\nChoose the wheel you want to spin!"
    kb = [[InlineKeyboardButton("Character Wheel 👤", callback_data="char_wheel")], [InlineKeyboardButton("Resource Wheel 💎", callback_data="res_wheel")]]
    await update.message.reply_video(WHEEL_VIDEO, caption=desc, reply_markup=InlineKeyboardMarkup(kb))

async def wheel_options(query, type_name):
    if type_name == "Character":
        cost1, cost5 = "150 🍀", "500 🍀"
        data_c1, data_c5 = "wheel_1_Character", "wheel_5_Character"
    else:
        cost1, cost5 = "100 🍀", "400 🍀"
        data_c1, data_c5 = "wheel_1_Resource", "wheel_5_Resource"

    desc = f"🎡 {type_name.upper()} WHEEL 🎡\n\n1x Pull: {cost1}\n5x Pull: {cost5}"
    kb = [[InlineKeyboardButton("1x Pull", callback_data=data_c1), InlineKeyboardButton("5x Pull", callback_data=data_c5)], [InlineKeyboardButton("Back", callback_data="wheel_cancel"), InlineKeyboardButton("Probability", callback_data="wheel_prob")]]
    await query.edit_message_media(InputMediaVideo(WHEEL_VIDEO, caption=desc), reply_markup=InlineKeyboardMarkup(kb))

async def handle_wheel(query, p, count, wheel_type):
    uid = str(p['user_id'])
    if wheel_type == "Character":
        cost = 150 if count == 1 else 500
    else:
        cost = 100 if count == 1 else 400

    if p.get("clovers", 0) < cost:
        await query.answer("Not enough 🍀 Clovers!", show_alert=True)
        return

    p["clovers"] -= cost
    save_player(uid, p)

    results = []
    special_anim = None

    if wheel_type == "Character":
        for _ in range(count):
            roll = random.random()
            
            # --- UPDATED PROBABILITIES ---
            # Ace, Law, Yamato, Kid: 2% Each
            if roll < 0.02: 
                res = "Portgas D Ace"
                special_anim = ACE_SUMMON_ANIM
            elif roll < 0.04:
                res = "Trafalgar D. Law"
                special_anim = LAW_SUMMON_ANIM
            elif roll < 0.06:
                res = "Yamato"
                special_anim = YAMATO_SUMMON_ANIM
            elif roll < 0.08:
                res = "Eustass Kid"
                special_anim = KID_SUMMON_ANIM
            elif roll < 0.18:
                # 10% Chance for Level Up Tokens on Character Wheel
                tokens = random.randint(1, 5)
                p['tokens'] += tokens
                results.append(f"🧩 {tokens} Level Up Token(s)!")
                continue
            else:
                # Exclude Specials
                excluded = ["Portgas D Ace", "Trafalgar D. Law", "Yamato", "Eustass Kid"]
                others = [c for c in CHARACTERS.keys() if c not in excluded]
                res = random.choice(others)

            char_data = CHARACTERS[res]
            rarity = char_data.get('rarity', 'Common')
            symbol = RARITY_STYLES.get(rarity, {}).get("symbol", "🔘")

            existing = next((c for c in p["characters"] if c["name"] == res), None)
            if existing:
                if existing.get("level", 1) < 40: # Max level capped at 40
                    existing["level"] = existing.get("level", 1) + 1
                    if existing["level"] > 40: existing["level"] = 40
                    results.append(f"• {res} {symbol} (Lv.{existing['level']})")
                else:
                    # Give berries instead if duplicate is max EXP level
                    bonus = 500
                    p['berries'] += bonus
                    results.append(f"• {res} {symbol} (Max Lvl! Converted to 🍇{bonus})")
            else:
                p["characters"].append(generate_char_instance(res))
                results.append(f"• {res} {symbol} (New!)")
    else:
        for _ in range(count):
            roll = random.random()
            if roll < 0.05:
                fruit_name = random.choice(list(DEVIL_FRUITS.keys()))
                p.setdefault("fruits", []).append(fruit_name)
                results.append(f"🍎 {fruit_name} (NEW!)")
            elif roll < 0.15: # 10% Chance for Level Up Token
                tokens = random.randint(1, 4)
                p['tokens'] += tokens
                results.append(f"🧩 {tokens} Level Up Token(s)!")
            elif roll < 0.35:
                clovers = random.randint(10, 50)
                p['clovers'] += clovers
                results.append(f"🍀 {clovers} Clovers")
            else:
                berries = random.randint(5000, 15000)
                p['berries'] += berries
                results.append(f"🍇 {berries} Berries")

    save_player(uid, p)
    res_text = f"🎰 **{wheel_type.upper()} RESULTS**:\n\n" + "\n".join(results)

    final_anim = special_anim if special_anim else SUMMON_ANIMATION
    try:
        await query.edit_message_media(InputMediaVideo(final_anim, caption=res_text, parse_mode="Markdown"), reply_markup=None)
    except Exception:
        await query.message.reply_text(res_text, parse_mode="Markdown")


# =====================
# INSPECT & FRUIT
# =====================

async def inspect_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/inspect [Name]`")
        return
    uid = str(update.effective_user.id)
    p = load_player(uid)
    if not load_player(update.effective_user.id):
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return

    name = " ".join(context.args).title()
    p = get_player(update.effective_user.id)

    if p and p.get('is_locked'):
        await update.effective_message.reply_text("❌ Your account is locked. Contact admin.")
        return

    if name in WEAPONS:
        w = WEAPONS[name]
        text = (f"➥Name: {name}\n➥Rarity: {w['rarity']}\n➥Attack: {w['atk_range']}\n"
                f"➥Critical chance: {w['crit']}\n➥Accuracy: {w['acc']}\n"
                f"➥Special attack: {w['spec']}\n➥Rank requirement: {w['lvl']}\n\n➥ Cost: {w['cost']}🍇")
        img = w['img']
        kb = [[InlineKeyboardButton("In Stock (Check /store) 🛒", callback_data="go_shop")]]
        await update.message.reply_photo(img, caption=text, reply_markup=InlineKeyboardMarkup(kb))
    elif name in DEVIL_FRUITS or any(k in name for k in ["Sand", "Shadow", "Barrier", "Munch", "Gum"]):
        search_name = name
        if "Sand" in name: search_name = "Sand Sand Fruit"
        elif "Shadow" in name: search_name = "Shadow Shadow Fruit"
        elif "Barrier" in name: search_name = "Barrier Barrier Fruit"
        elif "Munch" in name: search_name = "Munch Munch Fruit"
        elif "Gum" in name: search_name = "Gum Gum Fruit"

        if search_name not in DEVIL_FRUITS:
            await update.message.reply_text("Devil fruit not found.")
            return

        f = DEVIL_FRUITS[search_name]; kb = []
        if search_name in p.get("fruits", []):
            kb.append([InlineKeyboardButton("Equip (Consume) ⚡️", callback_data=f"equip_{search_name}")])
        else:
            kb.append([InlineKeyboardButton("In Stock (Check /store) 🛒", callback_data="go_shop")])

        await update.message.reply_photo(f['img'], caption=f['text'], reply_markup=InlineKeyboardMarkup(kb))
    else: await update.message.reply_text("Item not found.")

# =====================
# PROFILE & INV
# =====================

async def myprofile_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not load_player(update.effective_user.id):
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return
    uid = str(update.effective_user.id)
    p = load_player(uid)
    
    if p and p.get('is_locked'):
        await update.effective_message.reply_text("❌ Your account is locked. Contact admin.")
        return


    user_id = update.effective_user.id; p = get_player(user_id, update.effective_user.first_name)
    lvl = p.get('level', 1); exp = p.get('exp', 0); req = get_required_player_exp(lvl)
    wins = p.get('wins', 0); losses = p.get('losses', 0); total = wins + losses
    win_ratio = (wins / total * 100) if total > 0 else 0
    start_date = p.get('start_date', 'Unknown')
    safe_name = escape_md(p.get('name'))
    prof = f"⦿ Name: {safe_name}\n⦿ ID: {user_id}\n⦿ Level: {lvl}\n🌟 EXP: {exp}/{req}\n⦿ Bounty฿: {p.get('bounty', 100000):,}\n📅 Journey Started: {start_date}\n▰▱▱▱▱▱▱▱▱▱\n[ Ｓ Ｔ Ａ Ｔ Ｓ ]\n➜ 🏆 Victory: {wins}\n➜ 🏳️ Defeat: {losses}\n➜ 📊 Win Ratio: {win_ratio:.1f}%\n➜ ⚔️ Total Wins on explore: {p.get('explore_wins', 0)}"
    try:
        photos = await context.bot.get_user_profile_photos(user_id, limit=1)
        if photos.total_count > 0: await update.message.reply_photo(photos.photos[0][-1].file_id, caption=f"🏴‍☠️ **BOUNTY POSTER** 🏴‍☠️\n\n{prof}", parse_mode="Markdown")
        else: await update.message.reply_text(prof)
    except: await update.message.reply_text(prof)

async def inventory_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE, is_cb=False):
    if is_cb:
        uid = update.callback_query.from_user.id
    else:
        uid = update.effective_user.id
    
    if not load_player(uid):
        if is_cb:
            await update.callback_query.answer("⚠️ You must start your journey first!", show_alert=True)
        else:
            await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return

    p = get_player(uid)
    lvl = p.get('level', 1); exp = p.get('exp', 0); req = get_required_player_exp(lvl)
    safe_name = escape_md(p['name'])
    inv = f"╭━━━━━━━━━━━━━━━╮\n✦    📦 INVENTORY 📦     ✦\n╰━━━━━━━━━━━━━━━╯\n\nɴᴀᴍᴇ 📛: {safe_name}\nDevil fruit🪻: {p.get('equipped_fruit') or 'None'}\n━━━━━━━━━━━━━━━━━━━\nBerry🍇: {p.get('berries', 0):,}\nClover🍀: {p.get('clovers', 0):,}\n━━━━━━━━━━━━━━━━━━━\n\n━━━━━━━━━━━━━━━━━━━\nʟᴇᴠᴇʟ ⭐️: {lvl}\nᴇxᴘ 📈: {exp}/{req}\nʟᴠʟ ᴜᴘ ᴛᴏᴋᴇɴ 🧩: {p.get('tokens', 0)}\nᴋɪʟʟ ᴄᴏᴜɴᴛ 🩸: {p.get('kill_count', 0)}\n"
    
    # Updated Buttons with authorization check ID
    kb = [
        [InlineKeyboardButton("Weapons ⚔️", callback_data=f"inv_weapons|{uid}"), InlineKeyboardButton("Fruits 🍎", callback_data=f"inv_fruits|{uid}")],
        [InlineKeyboardButton("Artifacts 🏆", callback_data=f"inv_artifacts|{uid}"), InlineKeyboardButton("Keys 🗝", callback_data=f"inv_keys|{uid}")]
    ]

    if is_cb:
        await update.callback_query.edit_message_caption(caption=inv, reply_markup=InlineKeyboardMarkup(kb))
    else:
        await update.message.reply_photo(INVENTORY_IMAGE, caption=inv, reply_markup=InlineKeyboardMarkup(kb))

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not load_player(update.effective_user.id):
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return
    uid = str(update.effective_user.id)
    p = load_player(uid)
    if p and p.get('is_locked'):
        await update.effective_message.reply_text("❌ Your account is locked. Contact admin.")
        return


    p = get_player(update.effective_user.id)
    if not context.args: 
        await update.message.reply_text("⚠️ Usage: /stats [Character Name]")
        return
        
    input_name = " ".join(context.args)
    # Alias Check
    if input_name.lower() in NAME_ALIASES:
        input_name = NAME_ALIASES[input_name.lower()]

    # Check if user has this char
    char_obj = next((c for c in p.get('characters', []) if c['name'].lower() == input_name.lower()), None)
    
    if char_obj:
        await update.message.reply_photo(IMAGE_URLS.get(char_obj['name'], IMAGE_URLS["Default"]), caption=get_stats_text(char_obj, p.get('equipped_fruit')))
        return

    # Check global database
    found_key = next((k for k in CHARACTERS.keys() if k.lower() == input_name.lower()), None)
    if found_key:
        await update.message.reply_photo(IMAGE_URLS.get(found_key, IMAGE_URLS["Default"]), caption=get_stats_text(found_key, p.get('equipped_fruit')))
        return

    # Fuzzy match
    possible_matches = difflib.get_close_matches(input_name, CHARACTERS.keys(), n=1, cutoff=0.5)
    if possible_matches:
        await update.message.reply_text(f"Did you mean? {possible_matches[0]}")
    else:
        await update.message.reply_text("Character not found.")

async def mycollection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    p = get_player(user_id)
    if not p:
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return

    if p and p.get('is_locked'):
        await update.effective_message.reply_text("❌ Your account is locked. Contact admin.")
        return


    txt = "📜 *YOUR PIRATE FLEET* 📜\n━━━━━━━━━━━━━━━━━━━\n\n"
    
    if not p.get('characters'):
        txt += "_No pirates recruited yet._"
    else:
        for c in p['characters']:
            name = c['name']
            lvl = c.get('level', 1)
            
            char_master = CHARACTERS.get(name, {})
            rarity_type = char_master.get('rarity', 'Common')
            symbol = RARITY_STYLES.get(rarity_type, {}).get("symbol", "🔘")
            
            wep = f" | ⚔️ {c['equipped_weapon']}" if c.get('equipped_weapon') else ""
            txt += f"• *{name}* {symbol} (Lv.{lvl}){wep}\n"

    await update.message.reply_text(txt, parse_mode="Markdown")


async def sendberry_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not load_player(update.effective_user.id):
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return

    if not update.message.reply_to_message or not context.args: return
    try:
        amount = int(context.args[0]); sender_id = update.effective_user.id; receiver_id = update.message.reply_to_message.from_user.id; sender = get_player(sender_id)
        if sender.get('berries', 0) < amount or amount <= 0: return
        receiver = get_player(receiver_id, update.message.reply_to_message.from_user.first_name); sender['berries'] -= amount; receiver['berries'] += amount
        save_player(sender_id, sender); save_player(receiver_id, receiver); await update.message.reply_text(f"✅ Sent 🍇{amount:,} to {receiver['name']}")
    except: pass

async def sendclovers_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not load_player(update.effective_user.id):
        await update.message.reply_text("⚠️ You must start your journey first! Use /start.")
        return

    if not update.message.reply_to_message or not context.args: return
    try:
        amount = int(context.args[0]); sender_id = update.effective_user.id; receiver_id = update.message.reply_to_message.from_user.id; sender = get_player(sender_id)
        if sender.get('clovers', 0) < amount or amount <= 0: return
        receiver = get_player(receiver_id, update.message.reply_to_message.from_user.first_name); sender['clovers'] -= amount; receiver['clovers'] += amount
        save_player(sender_id, sender); save_player(receiver_id, receiver); await update.message.reply_text(f"✅ Sent 🍀{amount:,} to {receiver['name']}")
    except: pass

async def open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != constants.ChatType.PRIVATE:
        return

    user_id = update.effective_user.id
    if not get_player(user_id):
        await update.message.reply_text("⚠️ Start your journey first with /start.")
        return

    keyboard = [['Explore 🧭'], ['Close ❌']]
    markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    await update.message.reply_text(
        "🎮 *MENU OPENED* (DM Only)",
        reply_markup=markup,
        parse_mode="Markdown"
    )

async def close_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != constants.ChatType.PRIVATE:
        return

    await update.message.reply_text(
        "🔒 *MENU CLOSED*",
        reply_markup=ReplyKeyboardRemove()
    )

async def handle_menu_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != constants.ChatType.PRIVATE:
        return

    text = update.message.text
    
    if text == "Explore 🧭":
        return await explore_cmd(update, context)
        
    elif text == "Close ❌":
        return await close_cmd(update, context)

async def unstuck_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    p = load_player(uid)
    
    if not p:
        return

    if p.get('is_locked'):
        await update.message.reply_text("❌ You are currently locked by Marine Security. Contact Admin.")
        return

    p['last_interaction'] = 0 
    p['verification_active'] = False
    
    if uid in pending_explores:
        del pending_explores[uid]
    
    for bid in list(battles.keys()):
        if uid in bid:
            del battles[bid]
    
    save_player(uid, p)
    
    await update.message.reply_text("🛠 *System Reset!* Your session and battle timers have been unstuck.")
    
    await context.bot.send_message(
        chat_id="-1003855697962",
        text=f"🛠 *UNSTUCK:* User `{p.get('name', 'Unknown')}` (`{uid}`) reset their state.",
        parse_mode="Markdown"
    )

async def leaderboard_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if players_collection is None:
        await update.message.reply_text("❌ Database not connected.")
        return

    top_players = players_collection.find({}, {"name": 1, "bounty": 1}).sort("bounty", -1).limit(3)
    
    msg = "🏆 *BOUNTY LEADERBOARD* 🏆\n━━━━━━━━━━━━━━━━━━\n"
    
    count = 0
    for i, p in enumerate(top_players):
        medal = ["🥇", "🥈", "🥉"][i]
        msg += f"{medal} *{escape_md(p.get('name', 'Pirate'))}*\n   └─ ฿ {p.get('bounty', 0):,}\n"
        count += 1
        
    if count == 0:
        msg += "No pirates found."
        
    await update.message.reply_text(msg, parse_mode="Markdown")

async def dailytask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    p = load_player(uid)
    
    if not p:
        await update.message.reply_text("⚠️ Start your journey first!")
        return

    # Check Date Reset
    today = datetime.now().strftime("%Y-%m-%d")
    if "daily_stats" not in p or p["daily_stats"].get("date") != today:
        p["daily_stats"] = {
            "date": today,
            "kills": 0, "bounty_gained": 0, "clovers_gained": 0,
            "claimed": False
        }
        save_player(uid, p)

    stats = p["daily_stats"]
    
    # Tasks
    k_req, b_req, c_req = 50, 100000, 100
    
    k_mark = "✅" if stats["kills"] >= k_req else "❌"
    b_mark = "✅" if stats["bounty_gained"] >= b_req else "❌"
    c_mark = "✅" if stats["clovers_gained"] >= c_req else "❌"
    
    msg = (
        f"📅 *DAILY TASKS* ({today})\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{k_mark} Defeat 50 characters in Explore\n"
        f"   └─ {stats['kills']}/{k_req}\n\n"
        f"{b_mark} Gain 100,000 Bounty\n"
        f"   └─ {stats['bounty_gained']:,}/{b_req:,}\n\n"
        f"{c_mark} Gain 100 Clovers\n"
        f"   └─ {stats['clovers_gained']}/{c_req}\n"
    )
    
    kb = []
    if k_mark == "✅" and b_mark == "✅" and c_mark == "✅" and not stats["claimed"]:
        kb.append([InlineKeyboardButton("🎁 CLAIM REWARD (300 🍀)", callback_data="claim_daily")])
    elif stats["claimed"]:
        msg += "\n🎉 *REWARD CLAIMED!*"

    await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

# =====================
# NEW ADMIN COMMANDS
# =====================

async def add_char_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        return
        
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Usage: `/add <Character Name> <User ID>`")
        return
        
    # Last arg is ID, rest is name
    target_id = context.args[-1]
    char_name = " ".join(context.args[:-1])
    
    # Handle aliases
    if char_name.lower() in NAME_ALIASES:
        char_name = NAME_ALIASES[char_name.lower()]
    else:
        char_name = char_name.title()
    
    if char_name not in CHARACTERS:
        await update.message.reply_text(f"❌ Character '{char_name}' not found.")
        return
        
    p = load_player(target_id)
    if not p:
        await update.message.reply_text("❌ Player not found.")
        return
        
    p.setdefault('characters', [])
    
    # Check if they have it
    existing = next((c for c in p['characters'] if c['name'] == char_name), None)
    
    if existing:
        existing['level'] = 40
        existing['exp'] = get_required_char_exp(40)
        action = "updated to Level 40"
    else:
        new_char = generate_char_instance(char_name, level=40)
        p['characters'].append(new_char)
        action = "added at Level 40"
        
    save_player(target_id, p)
    await update.message.reply_text(f"✅ *{char_name}* {action} for user `{target_id}`.", parse_mode="Markdown")

async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        return

    if not context.args:
        await update.message.reply_text("⚠️ Usage: `/reset <user_id>`")
        return

    target_id = context.args[0]
    p = load_player(target_id)
    
    if not p:
        await update.message.reply_text("⚠️ Player not found.")
        return
        
    # Reset to default starter state but keep name/id
    default_p = {
        "user_id": str(target_id), 
        "name": p.get("name", "Pirate"), 
        "team": [], "characters": [],
        "berries": 10000, "clovers": 0, "bounty": 100000, "exp": 0, "level": 1,
        "starter_summoned": False, "wins": 0, "losses": 0, "explore_wins": 0, "kill_count": 0,
        "fruits": [], "equipped_fruit": None, "tokens": 0, "weapons": [],
        "artifacts": [], "keys": [],
        "explore_count": 0, "start_date": datetime.now().strftime("%Y-%m-%d"),
        "is_locked": False,
        "verification_active": False, "referred_by": None, "referrals": 0,
        "referral_reward_claimed": False,
        "daily_stats": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "kills": 0, "bounty_gained": 0, "clovers_gained": 0,
            "claimed": False
        }
    }
    
    save_player(target_id, default_p)
    await update.message.reply_text(f"✅ Player `{target_id}` has been fully reset to Level 1.", parse_mode="Markdown")

async def lock_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        return

    if not context.args:
        await update.message.reply_text("⚠️ Usage: `/lock <user_id>`")
        return
        
    target_id = context.args[0]
    p = load_player(target_id)
    
    if not p:
        await update.message.reply_text("⚠️ Player not found.")
        return
        
    p['is_locked'] = True
    save_player(target_id, p)
    await update.message.reply_text(f"🔒 Player `{target_id}` has been *LOCKED*.", parse_mode="Markdown")

async def info_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        return

    if not context.args:
        await update.message.reply_text("⚠️ Usage: `/info <user_id>`")
        return

    target_id = context.args[0]
    p = load_player(target_id)
    if not p:
        await update.message.reply_text("Player not found.")
        return

    # Inventory String
    inv_str = f"📦 *INVENTORY for {p['name']}*\n"
    inv_str += f"Berries: {p.get('berries',0)}\nClovers: {p.get('clovers',0)}\nTokens: {p.get('tokens',0)}\n"
    inv_str += f"Fruits: {', '.join(p.get('fruits', [])) or 'None'}\n"
    inv_str += f"Weapons: {', '.join(p.get('weapons', [])) or 'None'}\n"
    inv_str += f"Equipped Fruit: {p.get('equipped_fruit') or 'None'}"

    # Collection String
    col_str = f"📜 *COLLECTION for {p['name']}*\n"
    if not p.get('characters'):
        col_str += "No characters."
    else:
        for c in p['characters']:
            wep = f"(Wep: {c.get('equipped_weapon')})" if c.get('equipped_weapon') else ""
            col_str += f"• {c['name']} (Lv.{c['level']}) {wep}\n"

    await update.message.reply_text(inv_str)
    await update.message.reply_text(col_str)

async def users_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        return
    
    if players_collection:
        total = players_collection.count_documents({})
    else:
        total = "DB Error"
        
    online_approx = len(player_cache)
    
    await update.message.reply_text(
        f"📊 *USER STATISTICS*\n\n"
        f"🌍 Total Users: `{total}`\n"
        f"🟢 Online (RAM): `{online_approx}`",
        parse_mode="Markdown"
    )

async def crossover_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Comming soon...")

async def event_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("No events are avaliable right now.")

async def storymode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Comming soon...")

async def auto_detector_job(context: ContextTypes.DEFAULT_TYPE):
    current_time = time.time()

    for uid, p in list(player_cache.items()):
        last_act = p.get('last_interaction', 0)
        time_diff = current_time - last_act

        if (time_diff < 300) and not p.get('is_locked') and not p.get('verification_active'):
            if time_diff < 5:
                try:
                    await trigger_security_check(uid, context)
                    await asyncio.sleep(0.1) 
                except Exception as e:
                    logging.error(f"Security check failed for {uid}: {e}")


async def get_file_ids(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Only for Admins
    if update.effective_user.id not in ADMIN_IDS:
        return
        
    fid = update.message.photo[-1].file_id if update.message.photo else (update.message.video.file_id if update.message.video else None)
    if fid: await update.message.reply_text(f"File ID: `{fid}`", parse_mode="Markdown")

async def post_init(application):
    await application.bot.set_my_commands([
        BotCommand("start", "Start Journey"), 
        BotCommand("wheel", "Spin Wheel"), 
        BotCommand("explore", "Explore Grand Line"),
        BotCommand("myteam", "Manage Team"), 
        BotCommand("battle", "Challenge Player"), 
        BotCommand("stats", "Character Stats"),
        BotCommand("open", "Open Menu"), 
        BotCommand("close", "Close Menu"),
        BotCommand("mycollection", "View Crew"), 
        BotCommand("inventory", "Treasury"),
        BotCommand("myprofile", "Player Profile"), 
        BotCommand("unlock", "Unlock Player"),
        BotCommand("unstuck", "Reset Stuck Session"), 
        BotCommand("sendberry", "Gift Berries"), 
        BotCommand("sendclovers", "Gift Clovers"), 
        BotCommand("inspect", "Fruit/Weapon Info"),
        BotCommand("store", "Open Store"), 
        BotCommand("buy", "Buy Items"), 
        BotCommand("sell", "Sell Artifacts"),
        BotCommand("use", "Use Items"),
        BotCommand("referral", "Invite Friends"),
        BotCommand("crossover", "Crossover Event"),
        BotCommand("event", "Check Events"),
        BotCommand("storymode", "Story Mode"),
        BotCommand("leaderboard", "Top Bounty"),
        BotCommand("dailytask", "Daily Missions")
    ])

# =====================
# BOT EXECUTION
# =====================
TOKEN = "8308955773:AAHRa238qrLCJPcbBvgrSJk3EPqbiGsPsAI"

if __name__ == "__main__":

    if not TOKEN:
        print("❌ Error: BOT_TOKEN is missing!")
        exit(1)
    if not MONGO_URI:
        print("❌ Error: MONGO_URI is missing!")
        exit(1)

    application = ApplicationBuilder().token(TOKEN).post_init(post_init).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("open", open_cmd))
    application.add_handler(CommandHandler("close", close_cmd))
    application.add_handler(CommandHandler("wheel", wheel_cmd))
    application.add_handler(CommandHandler("explore", explore_cmd))
    application.add_handler(CommandHandler("stats", stats_cmd))
    application.add_handler(CommandHandler("unstuck", unstuck_cmd)) 
    application.add_handler(CommandHandler("unlock", unlock_cmd))   
    application.add_handler(CommandHandler("inspect", inspect_cmd))
    application.add_handler(CommandHandler("mycollection", mycollection))
    application.add_handler(CommandHandler("inventory", inventory_cmd))
    application.add_handler(CommandHandler("myprofile", myprofile_cmd))
    application.add_handler(CommandHandler("sendberry", sendberry_cmd))
    application.add_handler(CommandHandler("sendclovers", sendclovers_cmd))
    application.add_handler(CommandHandler("myteam", myteam))
    application.add_handler(CommandHandler("battle", battle_request))
    application.add_handler(CommandHandler("store", store_cmd))
    application.add_handler(CommandHandler("buy", buy_cmd))
    application.add_handler(CommandHandler("sell", sell_cmd))
    application.add_handler(CommandHandler("use", use_cmd))
    application.add_handler(CommandHandler("referral", referral_cmd))
    application.add_handler(CommandHandler("leaderboard", leaderboard_cmd))
    application.add_handler(CommandHandler("dailytask", dailytask_cmd))
    
    # New Commands
    application.add_handler(CommandHandler("reset", reset_cmd))
    application.add_handler(CommandHandler("lock", lock_cmd))
    application.add_handler(CommandHandler("info", info_cmd))
    application.add_handler(CommandHandler("add", add_char_cmd))
    application.add_handler(CommandHandler("users", users_cmd))
    application.add_handler(CommandHandler("crossover", crossover_cmd))
    application.add_handler(CommandHandler("event", event_cmd))
    application.add_handler(CommandHandler("storymode", storymode_cmd))
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_menu_click))
    application.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO, get_file_ids))
    application.add_handler(CallbackQueryHandler(main_callback))

    job_queue = application.job_queue
    job_queue.run_repeating(auto_detector_job, interval=900, first=10)

    print("🏴‍☠️ Pirate Bot is sailing with Marine Security Active!...")
    application.run_polling()