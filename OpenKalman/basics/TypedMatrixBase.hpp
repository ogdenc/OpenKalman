/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of TypedMatrixBase.
 */

#ifndef OPENKALMAN_TYPEDMATRIXBASE_HPP
#define OPENKALMAN_TYPEDMATRIXBASE_HPP

namespace OpenKalman::internal
{
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix, coefficients...Coefficients>
  requires (not std::is_rvalue_reference_v<NestedMatrix>) and (sizeof...(Coefficients) <= 2)
#else
  template<typename Derived, typename NestedMatrix, typename...Coefficients>
#endif
  struct TypedMatrixBase : MatrixBase<Derived, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert((coefficients<Coefficients> and ...));
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
    static_assert(sizeof...(Coefficients) <= 2);
#endif

  private:

    using Base = MatrixBase<Derived, NestedMatrix>;

  protected:

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar; ///< Scalar type for this variable.

  public:

    // -------------- //
    //  Constructors  //
    // -------------- //

    /// Default constructor.
#ifdef __cpp_concepts
    TypedMatrixBase() requires std::default_initializable<NestedMatrix> and (not dynamic_shape<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<
      std::is_default_constructible_v<T> and (not dynamic_shape<NestedMatrix>), int> = 0>
    TypedMatrixBase()
#endif
      : Base {} {}


    /// Construct from a typed_matrix_nestable.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (MatrixTraits<Arg>::rows == MatrixTraits<NestedMatrix>::rows) and
      (MatrixTraits<Arg>::columns == MatrixTraits<NestedMatrix>::columns) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (MatrixTraits<Arg>::rows == MatrixTraits<NestedMatrix>::rows) and
      (MatrixTraits<Arg>::columns == MatrixTraits<NestedMatrix>::columns) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    TypedMatrixBase(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a list of coefficients.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      requires(Args ... args) { NestedMatrix {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<
      (std::is_convertible_v<Args, const Scalar> and ...) and (sizeof...(Args) > 0) and
      ((diagonal_matrix<NestedMatrix> and std::is_constructible_v<NestedMatrix,
          typename MatrixTraits<NestedMatrix>::template NativeMatrixFrom<sizeof...(Args), 1>>) or
       (sizeof...(Args) == MatrixTraits<NestedMatrix>::rows * MatrixTraits<NestedMatrix>::columns and
          std::is_constructible_v<NestedMatrix,
          typename MatrixTraits<NestedMatrix>::template NativeMatrixFrom<MatrixTraits<NestedMatrix>::rows,
            MatrixTraits<NestedMatrix>::columns>>)), int> = 0>
#endif
    TypedMatrixBase(Args ... args)
      : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}


    // ---------------------- //
    //  Assignment Operators  //
    // ---------------------- //

    /// Assign from a \ref typed_matrix_nestable.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(arg));
        return *this;
      }
    }


    /// Multiply by a scale factor.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      this->nested_matrix() *= s;
      return *this;
    }


    /// Divide by a scale factor.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      this->nested_matrix() /= s;
      return *this;
    }

  };


} // namespace OpenKalman


#endif //OPENKALMAN_TYPEDMATRIXBASE_HPP
