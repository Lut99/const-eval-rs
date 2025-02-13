//  LIB.rs
//    by Lut99
//
//  Created:
//    13 Feb 2025, 13:03:08
//  Last edited:
//    13 Feb 2025, 14:52:18
//  Auto updated?
//    Yes
//
//  Description:
//!   <Todo>
//

// Modules
pub mod ir;

// Imports
use quote::ToTokens as _;
use syn::{BinOp, Expr, ExprBinary, ExprLit, Lit};

pub use crate::ir::Error;


/***** LIBRARY FUNCTIONS *****/
/// Evaluates the given Rust expression.
///
/// # Arguments
/// - `expr`: An [`Expr`] to evaluate.
///
/// # Returns
/// An [`ExprLit`] that represents the result of the expression.
///
/// Note that it will carry the span information of the `expr`ession as a whole.
///
/// # Errors
/// This function may return an error if the given expression type is not supported.
pub fn eval_expr(expr: &Expr) -> Result<ExprLit, Error> {
    // Switch to the correct evaluation function
    match expr {
        Expr::Binary(b) => eval_binary(b),
        Expr::Lit(l) => eval_lit(l),
        Expr::Paren(p) => eval_paren(p),
        Expr::Unary(u) => eval_unary(u),

        // The rest are (currently) unsupported expressions
        expr => Err(Error::UnsupportedExpr { expr: expr.to_token_stream().to_string() }),
    }
}



/// Evaluates the given binary operator.
///
/// # Arguments
/// - `binary`: The [`ExprBinary`] to evaluate.
///
/// # Returns
/// An [`ExprLit`] that represents the result of the expression.
///
/// Note that it will carry the span information of the `expr`ession as a whole.
///
/// # Errors
/// This function may return an error if the given expression type is not supported.
pub fn eval_binary(binary: &ExprBinary) -> Result<ExprLit, Error> {
    // Evaluate the parts
    let lhs: ExprLit = eval_expr(&binary.left)?;
    let rhs: ExprLit = eval_expr(&binary.right)?;

    // Decide on the operator
    match binary.op {
        BinOp::Add(_) | BinOp::BitAnd(_) | BinOp::BitOr(_) | BinOp::BitXor(_) | BinOp::Div(_) | BinOp::Mul(_) | BinOp::Rem(_) => {
            // Assert they are both integers
            let lhs: i64 = match lhs.lit {
                Lit::Int(i) => i.base10_parse::<i64>().map_err(|err| Error::IllegalInt { lit: i, err })?,
                lit => return Err(Error::IncorrectType { got: lit.into(), exp: ExprType::Integer,  }),
            }

            todo!()
        },
    }
}
