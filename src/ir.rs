//  IR.rs
//    by Lut99
//
//  Created:
//    13 Feb 2025, 13:39:19
//  Last edited:
//    13 Feb 2025, 15:21:11
//  Auto updated?
//    Yes
//
//  Description:
//!   Defines a cute little intermediate representation for the parts of
//!   the Rust expression tree that we evaluate.
//

use std::convert::Infallible;
use std::error;
use std::fmt::{Display, Formatter, Result as FResult};
use std::ops::{BitOr, BitOrAssign};

use proc_macro2::TokenStream as TokenStream2;
use quote::ToTokens as _;
use syn::spanned::Spanned as _;



/***** ERRORS *****/
/// Defines the errors occurring in expression evaluation.
#[derive(Debug)]
pub enum Error {
    /// Encountered an unsupported float.
    IllegalFloat { lit: syn::LitFloat, err: syn::Error },
    /// Encountered an unsupported integer.
    IllegalInteger { lit: syn::LitInt, err: syn::Error },
    /// Type mismathc
    TypeMismatch { expr: Expr, got: ExprType, exp: ExprType },
    /// Found an unsupported binary operation.
    UnsupportedBinOp { op: syn::BinOp },
    /// Found an unsupported expression.
    UnsupportedExpr { expr: syn::Expr },
    /// Found an unsupported literal.
    UnsupportedLit { lit: syn::Lit },
    /// Found an unsupported type.
    UnsupportedType { ty: syn::Type },
    /// Found an unsupported unary operation.
    UnsupportedUnOp { op: syn::UnOp },
}
impl Error {
    /// Turns this Error into an error emittable by a procedural macro.
    ///
    /// # Returns
    /// A [`TokenStream2`] that encodes the error with source information.
    #[inline]
    fn into_compile_error(self) -> TokenStream2 { syn::Error::from(self).into_compile_error() }
}
impl Display for Error {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> FResult {
        match self {
            Self::IllegalFloat { lit, .. } => write!(f, "Illegal float {:?}", lit.to_token_stream().to_string()),
            Self::IllegalInteger { lit, .. } => write!(f, "Illegal integer {:?}", lit.to_token_stream().to_string()),
            Self::UnsupportedBinOp { op } => {
                write!(f, "Encountered unsupported binary operator {:?}", op.to_token_stream().to_string())
            },
            Self::UnsupportedExpr { expr } => write!(f, "Encountered unsupported expression {:?}", expr.to_token_stream().to_string()),
            Self::UnsupportedLit { lit } => write!(f, "Encountered unsupported literal {:?}", lit.to_token_stream().to_string()),
            Self::UnsupportedType { ty } => {
                write!(f, "Encountered unsupported type {:?}", ty.to_token_stream().to_string())
            },
            Self::UnsupportedUnOp { op } => {
                write!(f, "Encountered unsupported unary operator {:?}", op.to_token_stream().to_string())
            },
        }
    }
}
impl error::Error for Error {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::IllegalFloat { err, .. } => Some(err),
            Self::IllegalInteger { err, .. } => Some(err),
            Self::TypeMismatch
            | Self::UnsupportedBinOp { .. }
            | Self::UnsupportedExpr { .. }
            | Self::UnsupportedLit { .. }
            | Self::UnsupportedType { .. }
            | Self::UnsupportedUnOp { .. } => None,
        }
    }
}
impl From<Error> for syn::Error {
    #[inline]
    fn from(value: Error) -> Self {
        Self::new(
            match &value {
                Error::IllegalFloat { lit, .. } => lit.span(),
                Error::IllegalInteger { lit, .. } => lit.span(),
                Error::TypeMismatch { expr, .. } => expr.span(),
                Error::UnsupportedBinOp { op } => op.span(),
                Error::UnsupportedExpr { expr } => expr.span(),
                Error::UnsupportedLit { lit } => lit.span(),
                Error::UnsupportedType { ty } => ty.span(),
                Error::UnsupportedUnOp { op } => op.span(),
            },
            value.to_string(),
        )
    }
}





/***** AUXILLARY *****/
/// Supported data types.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExprType {
    /// Whole numbers (both signed- and unsigned)
    Integer = 0x01,
    /// Decimal numbers (both signed- and unsigned)
    Float = 0x02,
}
impl ExprType {
    /// Constant-time evaluable conversion of this type into a unique bit to represent it.
    ///
    /// # Returns
    /// An [`u8`] to represent this type.
    #[inline]
    pub const fn as_u8(&self) -> u8 {
        match self {
            ExprType::Integer => 0x01,
            ExprType::Float => 0x02,
        }
    }
}
impl BitOr for ExprType {
    type Output = ExprTypeFilter;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output { ExprTypeFilter(u8::from(self) | u8::from(rhs)) }
}
impl From<u8> for ExprType {
    #[inline]
    fn from(value: u8) -> Self {
        match value {
            0x01 => Self::Integer,
            0x02 => Self::Float,
            other => panic!("Encountered illegal expression type integer {other:?}"),
        }
    }
}
impl From<ExprType> for u8 {
    #[inline]
    fn from(value: ExprType) -> Self { value.as_u8() }
}
impl TryFrom<syn::Type> for ExprType {
    type Error = Error;

    #[inline]
    fn try_from(value: syn::Type) -> Result<Self, Self::Error> {
        match value {
            syn::Type::Path(p) => p.try_into(),
            ty => Err(Error::UnsupportedType { ty }),
        }
    }
}
impl TryFrom<syn::TypePath> for ExprType {
    type Error = Error;

    #[inline]
    fn try_from(value: syn::TypePath) -> Result<Self, Self::Error> {
        // Get the identifier (either the path directly, or a reference to the core)
        let ident: &syn::Ident = if let Some(ident) = value.path.get_ident() {
            ident
        } else {
            if value.path.leading_colon.is_none() {
                return Err(Error::UnsupportedType { ty: syn::Type::Path(value) });
            }
            let mut ident: Option<&syn::Ident> = None;
            for (i, seg) in value.path.segments.iter().enumerate() {
                if i == 0 && (seg.ident == "core" || seg.ident == "std") {
                    continue;
                } else if i == 1 {
                    ident = Some(&seg.ident);
                } else {
                    return Err(Error::UnsupportedType { ty: syn::Type::Path(value) });
                }
            }
            match ident {
                Some(ident) => ident,
                None => return Err(Error::UnsupportedType { ty: syn::Type::Path(value) }),
            }
        };

        // See if it's one we want
        if ident == "i64" {
            Ok(Self::Integer)
        } else if ident == "f64" {
            Ok(Self::Float)
        } else {
            Err(Error::UnsupportedType { ty: syn::Type::Path(value) })
        }
    }
}

/// Explains filters of expression types.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExprTypeFilter(u8);
impl ExprTypeFilter {
    /// Represents a filter that matches all types.
    pub const ALL: Self = Self(0xFF);
    /// Represents a filter that matches no types.
    pub const NONE: Self = Self(0x00);



    /// Checks whether the given expression type matches the filter.
    ///
    /// # Arguments
    /// - `ty`: The [`ExprType`] to match the filter.
    ///
    /// # Returns
    /// True if it's filtered by this filter, or false otherwise.
    #[inline]
    pub fn matches(&self, ty: ExprType) -> bool {
        let ty_int: u8 = ty.into();
        self.0 & ty_int == ty_int
    }
}
impl BitOr<ExprType> for ExprTypeFilter {
    type Output = ExprTypeFilter;

    #[inline]
    fn bitor(self, rhs: ExprType) -> Self::Output { Self(self.0 | u8::from(rhs)) }
}
impl BitOrAssign<ExprType> for ExprTypeFilter {
    #[inline]
    fn bitor_assign(&mut self, rhs: ExprType) { self.0 |= u8::from(rhs); }
}
impl From<ExprType> for ExprTypeFilter {
    #[inline]
    fn from(value: ExprType) -> Self { Self(value.into()) }
}

/// Explains the expected types of a binary operator's side.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExprBinopFilter {
    /// Expect the same thing on both sides.
    Equal(ExprTypeFilter),
    /// Expect two different types.
    Diff(ExprTypeFilter, ExprTypeFilter),
}
impl ExprBinopFilter {
    /// Checks if the given two types match this filter.
    ///
    /// # Arguments
    /// - `lhs`: The [`ExprType`] of the lefthand-side to check.
    /// - `rhs`: The [`ExprType`] of the righthand-side to check.
    ///
    /// # Returns
    /// True if the given types are matches for this filter, or false otherwise.
    #[inline]
    pub fn matches(&self, lhs: ExprType, rhs: ExprType) -> bool {
        match self {
            Self::Equal(filter) => filter.matches(lhs) && filter.matches(rhs),
            Self::Diff(lhs_filter, rhs_filter) => lhs_filter.matches(lhs) && rhs_filter.matches(rhs),
        }
    }
}





/***** LIBRARY *****/
/// Defines the toplevel expression-type.
#[derive(Clone, Debug)]
pub enum Expr {
    /// It's a binary operator.
    BinOp(ExprBinOp),
    /// It's a unary operator.
    UnaOp(ExprUnaOp),
    /// It's a cast operator.
    Cast(ExprCast),
    /// It's a literal.
    Lit(ExprLit),
}
impl TryFrom<syn::Expr> for Expr {
    type Error = Error;

    #[inline]
    fn try_from(value: syn::Expr) -> Result<Self, Self::Error> {
        match value {
            syn::Expr::Binary(b) => Ok(Self::BinOp(b.try_into()?)),
            syn::Expr::Unary(u) => Ok(Self::UnaOp(u.try_into()?)),
            syn::Expr::Cast(c) => Ok(Self::Cast(c.try_into()?)),
            syn::Expr::Lit(l) => Ok(Self::Lit(l.try_into()?)),
            expr => Err(Error::UnsupportedExpr { expr }),
        }
    }
}



/// Defines binary operations in expressions.
#[derive(Clone, Debug)]
pub struct ExprBinOp {
    /// The operator to apply.
    pub op:  BinOp,
    /// The lefthand-side of the expression.
    pub lhs: Box<Expr>,
    /// The righthand-side of the expression.
    pub rhs: Box<Expr>,
}
impl ExprBinOp {
    /// Computes the evaluation of this node.
    ///
    /// # Returns
    /// An [`ExprLit`] with the result of the operation.
    #[inline]
    pub const fn eval(&self) -> Result<ExprLit, Error> {
        // Evaluate the left- and righthand-side
        let (lhs, rhs): (ExprLit, ExprLit) = (self.lhs.eval()?, self.rhs.eval()?);

        // Run the op
        match self.op {
            BinOp::Add => match (lhs, rhs) {
                (ExprLit::Integer(lhs), ExprLit::Integer(rhs)) => Ok(ExprLit::Integer(lhs + rhs)),
                (ExprLit::Integer(_), ExprLit::Float(_)) => {
                    Err(Error::TypeMismatch { expr: self.rhs.clone(), got: ExprType::Float, exp: ExprType::Integer })
                },
                (ExprLit::Float(lhs), ExprLit::Integer(rhs)) => {
                    Err(Error::TypeMismatch { expr: self.rhs.clone(), got: ExprType::Integer, exp: ExprType::Float })
                },
                (ExprLit::Float(lhs), ExprLit::Float(rhs)) => Ok(ExprLit::Float(lhs + rhs)),
            },
        }
    }

    /// Evaluates the type of this unary operator.
    ///
    /// # Returns
    /// An [`ExprType`] with the type of the expression after evaluation.
    #[inline]
    pub const fn ty(&self) -> ExprType {
        match self.op {
            UnaOp::Neg => self.expr.ty(),
        }
    }
}
impl TryFrom<syn::ExprBinary> for ExprBinOp {
    type Error = Error;

    #[inline]
    fn try_from(value: syn::ExprBinary) -> Result<Self, Self::Error> {
        Ok(Self { op: value.op.try_into()?, lhs: Box::new((*value.left).try_into()?), rhs: Box::new((*value.right).try_into()?) })
    }
}

/// Defines the actually supported operators.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BinOp {
    // Arithmetic
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplcation
    Mul,
    /// Division
    Div,
    /// Remainder
    Mod,

    // Binary
    /// Bitwise and
    BitAnd,
    /// Bitwise or
    BitOr,
    /// Bitwise xor
    BitXor,
    /// Leftshift
    LSh,
    /// Rightshift
    RSh,
}
impl BinOp {
    /// Returns the allowed types for each of the sides of this expression.
    ///
    /// # Returns
    /// An [`ExprBinopFilter`] that can match allowed types for this operator.
    pub fn allowed_types(&self) -> ExprBinopFilter {
        match self {
            Self::Add => ExprBinopFilter::Equal(ExprType::Integer | ExprType::Float),
            Self::Sub => ExprBinopFilter::Equal(ExprType::Integer | ExprType::Float),
            Self::Mul => ExprBinopFilter::Equal(ExprType::Integer | ExprType::Float),
            Self::Div => ExprBinopFilter::Equal(ExprType::Integer | ExprType::Float),
            Self::Mod => ExprBinopFilter::Equal(ExprType::Integer.into()),

            Self::BitAnd => ExprBinopFilter::Equal(ExprTypeFilter::ALL),
            Self::BitOr => ExprBinopFilter::Equal(ExprTypeFilter::ALL),
            Self::BitXor => ExprBinopFilter::Equal(ExprTypeFilter::ALL),
            Self::LSh => ExprBinopFilter::Diff(ExprTypeFilter::ALL, ExprType::Integer.into()),
            Self::RSh => ExprBinopFilter::Diff(ExprTypeFilter::ALL, ExprType::Integer.into()),
        }
    }
}
impl TryFrom<syn::BinOp> for BinOp {
    type Error = Error;

    #[inline]
    fn try_from(value: syn::BinOp) -> Result<Self, Self::Error> {
        match value {
            syn::BinOp::Add(_) => Ok(Self::Add),
            syn::BinOp::Sub(_) => Ok(Self::Sub),
            syn::BinOp::Mul(_) => Ok(Self::Mul),
            syn::BinOp::Div(_) => Ok(Self::Div),
            syn::BinOp::Rem(_) => Ok(Self::Mod),

            syn::BinOp::BitAnd(_) => Ok(Self::BitAnd),
            syn::BinOp::BitOr(_) => Ok(Self::BitOr),
            syn::BinOp::BitXor(_) => Ok(Self::BitXor),
            syn::BinOp::Shl(_) => Ok(Self::LSh),
            syn::BinOp::Shr(_) => Ok(Self::RSh),

            op => Err(Error::UnsupportedBinOp { op }),
        }
    }
}



/// Defines unary operations in expressions.
#[derive(Clone, Debug)]
pub struct ExprUnaOp {
    /// The operator to apply.
    pub op:   UnaOp,
    /// The expression to apply it to.
    pub expr: Box<Expr>,
}
impl ExprUnaOp {
    /// Computes the evaluation of this node.
    ///
    /// # Returns
    /// An [`ExprLit`] with the result of the operation.
    #[inline]
    pub const fn eval(&self) -> Result<ExprLit, Error> {
        // Evaluate the expression
        let expr: ExprLit = self.expr.eval()?;

        // Run the op
        match self.op {
            UnaOp::Neg => match expr {
                ExprLit::Integer(i) => Ok(ExprLit::Integer(-i)),
                ExprLit::Float(i) => Ok(ExprLit::Float(-i)),
            },
        }
    }

    /// Evaluates the type of this unary operator.
    ///
    /// # Returns
    /// An [`ExprType`] with the type of the expression after evaluation.
    #[inline]
    pub const fn ty(&self) -> ExprType {
        match self.op {
            UnaOp::Neg => self.expr.ty(),
        }
    }
}
impl TryFrom<syn::ExprUnary> for ExprUnaOp {
    type Error = Error;
    #[inline]
    fn try_from(value: syn::ExprUnary) -> Result<Self, Self::Error> {
        Ok(Self { op: value.op.try_into()?, expr: Box::new((*value.expr).try_into()?) })
    }
}

/// Defines the actually supported operators.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UnaOp {
    /// Arithmetic negation.
    Neg,
}
impl UnaOp {
    /// Returns the allowed types for the expression.
    ///
    /// # Returns
    /// An [`ExprTypeFilter`] that can match allowed types for this operator.
    pub fn allowed_types(&self) -> ExprTypeFilter {
        match self {
            Self::Neg => ExprType::Integer | ExprType::Float,
        }
    }
}
impl TryFrom<syn::UnOp> for UnaOp {
    type Error = Error;

    #[inline]
    fn try_from(value: syn::UnOp) -> Result<Self, Self::Error> {
        match value {
            syn::UnOp::Neg(_) => Ok(Self::Neg),
            op => Err(Error::UnsupportedUnOp { op }),
        }
    }
}



/// Defines a casting expression.
#[derive(Clone, Debug)]
pub struct ExprCast {
    /// The type to cast to.
    pub to:   ExprType,
    /// The expression who's result we're casting.
    pub expr: Box<Expr>,
}
impl ExprCast {
    /// Computes the evaluation of this cast node.
    ///
    /// # Returns
    /// An [`ExprLit`] with the casted value.
    #[inline]
    pub const fn eval(&self) -> Result<ExprLit, Error> {
        // Evaluate the expression
        let res: ExprLit = self.expr.eval()?;

        // Attempt to cast
        match (res, self.to) {
            (ExprLit::Integer(i), ExprType::Integer) => Ok(ExprLit::Integer(i)),
            (ExprLit::Integer(i), ExprType::Float) => Ok(ExprLit::Float(i as f64)),
            (ExprLit::Float(f), ExprType::Integer) => Ok(ExprLit::Integer(f as i64)),
            (ExprLit::Float(f), ExprType::Float) => Ok(ExprLit::Float(f)),
        }
    }

    /// Evaluates the type of this casting operator.
    ///
    /// # Returns
    /// An [`ExprType`] with the type we cast to.
    #[inline]
    pub const fn ty(&self) -> ExprType { self.to }
}
impl TryFrom<syn::ExprCast> for ExprCast {
    type Error = Error;

    #[inline]
    fn try_from(value: syn::ExprCast) -> Result<Self, Self::Error> {
        Ok(Self { to: (*value.ty).try_into()?, expr: Box::new((*value.expr).try_into()?) })
    }
}



/// Defines a literal expression
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExprLit {
    /// Integers
    Integer(i64),
    /// Floating-point
    Float(f64),
}
impl ExprLit {
    /// Computes the evaluation of this expression node.
    ///
    /// # Returns
    /// An [`ExprLit`] with this literal's value.
    #[inline]
    pub const fn eval(&self) -> ExprLit { *self }

    /// Evaluates the type of this literal.
    ///
    /// # Returns
    /// An [`ExprType`] with the type of this literal.
    #[inline]
    pub const fn ty(&self) -> ExprType {
        match self {
            Self::Integer(_) => ExprType::Integer,
            Self::Float(_) => ExprType::Float,
        }
    }
}
impl TryFrom<syn::ExprLit> for ExprLit {
    type Error = Error;

    #[inline]
    fn try_from(value: syn::ExprLit) -> Result<Self, Self::Error> {
        match value.lit {
            syn::Lit::Int(i) => Ok(Self::Integer(i.base10_parse().map_err(|err| Error::IllegalInteger { lit: i, err })?)),
            syn::Lit::Float(f) => Ok(Self::Float(f.base10_parse().map_err(|err| Error::IllegalFloat { lit: f, err })?)),
            lit => Err(Error::UnsupportedLit { lit }),
        }
    }
}
