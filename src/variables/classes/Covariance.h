/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COVARIANCE_H
#define OPENKALMAN_COVARIANCE_H

#include <initializer_list>
#include "variables/support/CovarianceBase.h"

namespace OpenKalman
{
  //////////////////
  //  Covariance  //
  //////////////////

  template<typename Coeffs, typename ArgType>
  struct Covariance
    : internal::CovarianceBase<Covariance<Coeffs, ArgType>, ArgType>
  {
    static_assert(OpenKalman::is_covariance_base_v<ArgType>);
    using BaseMatrix = ArgType;
    using Coefficients = Coeffs;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    using Base = internal::CovarianceBase<Covariance, ArgType>;

  protected:
    static constexpr TriangleType storage_type =
      OpenKalman::triangle_type_of_v<typename MatrixTraits<BaseMatrix>::template TriangularBaseType<>>;

    using SABaseType = std::conditional_t<is_diagonal_v<BaseMatrix>, BaseMatrix,
      typename MatrixTraits<BaseMatrix>::template SelfAdjointBaseType<storage_type>>;

    template<typename C = Coefficients, typename Arg>
    static constexpr auto
    make(Arg&& arg) noexcept
    {
      return Covariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

  public:
    /**************
     * Constructors
     **************/

    /// Default constructor.
    Covariance() : Base() {}

    /// Copy constructor.
    Covariance(const Covariance& other) : Base(other.base_matrix()) {}

    /// Move constructor.
    Covariance(Covariance&& other) noexcept : Base(std::move(other).base_matrix()) {}

    /// Convert from a general covariance type.
    template<typename M, std::enable_if_t<is_covariance_v<M>, int> = 0>
    Covariance(M&& m) noexcept : Base(internal::convert_base_matrix<BaseMatrix>(std::forward<M>(m)))
    {
      static_assert(is_equivalent_v<typename MatrixTraits<M>::Coefficients, Coefficients>);
    }

    /// Construct from a covariance base, general case.
    template<typename M, std::enable_if_t<is_covariance_base_v<M> and
      ((is_diagonal_v<M> or is_diagonal_v<BaseMatrix> or is_upper_triangular_v<M> == is_upper_triangular_v<BaseMatrix>) and
      (not is_diagonal_v<M> or is_self_adjoint_v<BaseMatrix>)), int> = 0>
    Covariance(M&& m) noexcept : Base(std::forward<M>(m)) {}

    /// Construct from a covariance base, if it has a different triangle type.
    template<typename M, std::enable_if_t<is_covariance_base_v<M> and
      not is_diagonal_v<M> and not is_diagonal_v<BaseMatrix> and is_upper_triangular_v<M> != is_upper_triangular_v<BaseMatrix>, int> = 0>
    Covariance(M&& m) noexcept : Base(adjoint(std::forward<M>(m))) {}

    /// Construct from a covariance base, diagonal case that needs to be squared.
    template<typename M, std::enable_if_t<is_covariance_base_v<M> and
      is_diagonal_v<M> and not is_self_adjoint_v<BaseMatrix>, int> = 0>
    Covariance(M&& m) noexcept : Base(Cholesky_factor(std::forward<M>(m))) {}

    /// Construct from a typed matrix (assumed to be self-adjoint).
    template<typename M, std::enable_if_t<is_typed_matrix_v<M>, int> = 0>
    Covariance(M&& m) noexcept : Base(MatrixTraits<SABaseType>::make(OpenKalman::base_matrix(std::forward<M>(m))))
    {
      static_assert(is_equivalent_v<typename MatrixTraits<M>::RowCoefficients, Coefficients>);
      if constexpr(not is_diagonal_v<BaseMatrix>)
        static_assert(is_equivalent_v<typename MatrixTraits<M>::ColumnCoefficients, Coefficients>);
    }

    /// Construct from a typed matrix base (assumed to be self-adjoint).
    template<typename M, std::enable_if_t<is_typed_matrix_base_v<M> and not is_covariance_base_v<M>, int> = 0>
    Covariance(M&& m) noexcept : Base(MatrixTraits<SABaseType>::make(std::forward<M>(m))) {}

    /// Construct from Scalar coefficients. Assumes matrix is self-adjoint, and only reads lower left triangle.
    template<typename ... Args, std::enable_if_t<
      std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
    Covariance(Args ... args) : Base(MatrixTraits<SABaseType>::make(args...)) {}

    /**********************
     * Assignment Operators
     **********************/

    /// Copy assignment operator.
    auto& operator=(const Covariance& other)
    {
      if (this != &other) base_matrix() = other.base_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(Covariance&& other) noexcept
    {
      if (this != &other) base_matrix() = std::move(other).base_matrix();
      return *this;
    }

    /// Assign from a compatible covariance type.
    template<typename Arg, std::enable_if_t<is_covariance_v<Arg>, int> = 0>
    auto& operator=(Arg&& other) noexcept
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      if constexpr (std::is_same_v<std::decay_t<Arg>, Covariance>) if (this == &other) return *this;
      base_matrix() = internal::convert_base_matrix<BaseMatrix>(std::forward<Arg>(other));
      return *this;
    }

    template<typename Arg, std::enable_if_t<is_covariance_v<Arg>, int> = 0>
    auto& operator+=(Arg&& arg)
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      static_assert(not is_square_root_v<Arg>);
      if constexpr(is_self_adjoint_v<BaseMatrix>)
      {
        base_matrix() += internal::convert_base_matrix<BaseMatrix>(std::forward<Arg>(arg));
      }
      else
      {
        decltype(auto) E1 = base_matrix();
        decltype(auto) E2 = internal::convert_base_matrix<BaseMatrix>(std::forward<Arg>(arg));
        if constexpr(is_upper_triangular_v<BaseMatrix>)
          base_matrix() = QR_decomposition(concatenate_vertical(E1, E2));
        else
          base_matrix() = LQ_decomposition(concatenate_horizontal(E1, E2));
      }
      return *this;
    }

    template<typename Arg, std::enable_if_t<is_covariance_v<Arg>, int> = 0>
    auto& operator-=(Arg&& arg)
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      static_assert(not is_square_root_v<Arg>);
      if constexpr(is_self_adjoint_v<BaseMatrix>)
      {
        base_matrix() -= internal::convert_base_matrix<BaseMatrix>(std::forward<Arg>(arg));
      }
      else
      {
        using TLowerType = typename MatrixTraits<BaseMatrix>::template TriangularBaseType<TriangleType::lower>;
        const auto U = internal::convert_base_matrix<TLowerType>(std::forward<Arg>(arg));
        rank_update(base_matrix(), U, Scalar(-1));
      }
      return *this;
    }

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator*=(const S s)
    {
      if constexpr(is_self_adjoint_v<BaseMatrix>)
      {
        base_matrix() *= s;
      }
      else
      {
        if (s > S(0))
        {
          base_matrix() *= std::sqrt(static_cast<Scalar>(s));
        }
        else if (s < S(0))
        {
          using TLowerType = typename MatrixTraits<BaseMatrix>::template TriangularBaseType<TriangleType::lower>;
          const auto U = internal::convert_base_matrix<TLowerType>(*this);
          base_matrix() = MatrixTraits<BaseMatrix>::zero();
          rank_update(base_matrix(), U, s);
        }
        else
        {
          base_matrix() = MatrixTraits<BaseMatrix>::zero();
        }
      }
      return *this;
    }

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator/=(const S s)
    {
      if constexpr(OpenKalman::is_self_adjoint_v<BaseMatrix>)
      {
        base_matrix() /= s;
      }
      else
      {
        if (s > S(0))
        {
          base_matrix() /= std::sqrt(static_cast<Scalar>(s));
        }
        else if (s < S(0))
        {
          using TLowerType = typename MatrixTraits<BaseMatrix>::template TriangularBaseType<TriangleType::lower>;
          const auto u = internal::convert_base_matrix<TLowerType>(*this);
          base_matrix() = MatrixTraits<BaseMatrix>::zero();
          rank_update(base_matrix(), u, 1 / static_cast<Scalar>(s));
        }
        else
        {
          throw (std::runtime_error("Covariance operator/=: divide by zero"));
        }
      }
      return *this;
    }

    /// Scale by a factor. Equivalent to multiplication by the square of a scalar.
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto&
    scale(const S s)
    {
      if constexpr(is_self_adjoint_v<BaseMatrix>)
        base_matrix() *= static_cast<Scalar>(s) * s;
      else
        base_matrix() *= s;
      return *this;
    }

    /// Scale by the inverse of a scalar factor. Equivalent by division by the square of a scalar.
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto&
    inverse_scale(const S s)
    {
      if constexpr(is_self_adjoint_v<BaseMatrix>)
        base_matrix() /= static_cast<Scalar>(s) * s;
      else
        base_matrix() /= s;
      return *this;
    }


    /*********
     * Other
     *********/

    static auto zero() { return make(MatrixTraits<BaseMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<BaseMatrix>::identity()); }

    using Base::base_matrix;

  };


  /////////////////////////////////////
  //        Deduction guides         //
  /////////////////////////////////////

  template<typename M, std::enable_if_t<is_covariance_v<M>, int> = 0>
  Covariance(M&&) -> Covariance<typename MatrixTraits<M>::Coefficients, typename MatrixTraits<M>::BaseMatrix>;

  template<typename M, std::enable_if_t<is_covariance_base_v<M>, int> = 0>
  Covariance(M&&) -> Covariance<Axes<MatrixTraits<M>::dimension>, std::decay_t<M>>;

  template<typename M, std::enable_if_t<is_typed_matrix_v<M>, int> = 0>
  Covariance(M&&) -> Covariance<
    typename MatrixTraits<M>::RowCoefficients,
    typename MatrixTraits<typename MatrixTraits<M>::BaseMatrix>::template SelfAdjointBaseType<>>;

  template<typename M, std::enable_if_t<is_typed_matrix_base_v<M> and not is_covariance_base_v<M>, int> = 0>
  Covariance(M&&) -> Covariance<
    Axes<MatrixTraits<M>::dimension>,
    typename MatrixTraits<M>::template SelfAdjointBaseType<>>;


  //////////////////////
  //  Make Functions  //
  //////////////////////

  // Make from covariance base or regular matrix:

  /// Make a Covariance based on a covariance base.
  template<typename Coefficients, typename Arg, std::enable_if_t<is_covariance_base_v<Arg>, int> = 0>
  inline auto
  make_Covariance(Arg&& arg) noexcept
  {
    return Covariance<Coefficients, std::decay_t<Arg>>(std::forward<Arg>(arg));
  }


  /// Make a Covariance, converting from a matrix other than a covariance base.
  template<typename Coefficients, TriangleType ... triangle_type, typename Arg,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and not is_covariance_base_v<Arg> and
      is_typed_matrix_base_v<Arg>, int> = 0>
  inline auto
  make_Covariance(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    constexpr TriangleType arg_t_type = triangle_type_of_v<typename MatrixTraits<Arg>::template TriangularBaseType<>>;
    constexpr TriangleType t_type = (arg_t_type, ... , triangle_type);
    using T = typename MatrixTraits<Arg>::template TriangularBaseType<t_type>;
    using SA = typename MatrixTraits<Arg>::template SelfAdjointBaseType<t_type>;
    using B = std::conditional_t<sizeof...(triangle_type) == 1, T, SA>;
    const SA b = std::forward<Arg>(arg);
    return Covariance<Coefficients, B> {b};
  }


  /// Make an axes-only Covariance, based on a covariance base or regular matrix.
  template<TriangleType ... triangle_type, typename Arg, std::enable_if_t<
    ((sizeof...(triangle_type)) == 0 and is_covariance_base_v<Arg>) or
    ((sizeof...(triangle_type)) <= 1 and is_typed_matrix_base_v<Arg>), int> = 0>
  inline auto
  make_Covariance(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_Covariance<C, triangle_type...>(std::forward<Arg>(arg));
  }


  /// Make a default Covariance, based on a template type, for a covariance base or regular matrix.
  template<typename Coefficients, typename Arg,
    std::enable_if_t<is_covariance_base_v<Arg> or is_typed_matrix_base_v<Arg>, int> = 0>
  inline auto
  make_Covariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    constexpr TriangleType triangle_type = triangle_type_of_v<typename MatrixTraits<Arg>::template TriangularBaseType<>>;

    using B = std::conditional_t<OpenKalman::is_diagonal_v<Arg>,
      typename MatrixTraits<Arg>::template DiagonalBaseType<>,
      std::conditional_t<is_triangular_v<Arg>,
        typename MatrixTraits<Arg>::template TriangularBaseType<triangle_type>,
        typename MatrixTraits<Arg>::template SelfAdjointBaseType<triangle_type>>>;

    return Covariance<Coefficients, B>();
  }


  /// Make a default Covariance, based on a template type, for a regular matrix.
  template<typename Coefficients, TriangleType triangle_type, typename Arg,
    std::enable_if_t<is_typed_matrix_base_v<Arg>, int> = 0>
  inline auto
  make_Covariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);

    using B = std::conditional_t<is_diagonal_v<Arg>,
      typename MatrixTraits<Arg>::template DiagonalBaseType<>,
      typename MatrixTraits<Arg>::template TriangularBaseType<triangle_type>>;

    return Covariance<Coefficients, B>();
  }


  /// Make a default axes-only Covariance, based on a template type.
  template<typename Arg, std::enable_if_t<is_covariance_base_v<Arg> or is_typed_matrix_base_v<Arg>, int> = 0>
  inline auto
  make_Covariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_Covariance<C, Arg>();
  }


  /// Make a default axes-only Covariance for a regular matrix.
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<is_typed_matrix_base_v<Arg>, int> = 0>
  inline auto
  make_Covariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_Covariance<C, triangle_type, Arg>();
  }


  // Make from another covariance type

  /// Make a Covariance based on another covariance.
  template<typename Arg, std::enable_if_t<is_covariance_v<Arg>, int> = 0>
  inline auto
  make_Covariance(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    return make_Covariance<C>(base_matrix(std::forward<Arg>(arg)));
  }


  /// Make a default Covariance, based on a covariance template type.
  template<typename Arg, std::enable_if_t<is_covariance_v<Arg>, int> = 0>
  inline auto
  make_Covariance()
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    using B = typename MatrixTraits<Arg>::BaseMatrix;
    return make_Covariance<C, B>();
  }


  // Make from a typed matrix

  /// Make a Covariance from a typed matrix.
  template<TriangleType...triangle_type, typename Arg,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and is_typed_matrix_v<Arg>, int> = 0>
  inline auto
  make_Covariance(Arg&& arg) noexcept
  {
    static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    return make_Covariance<C, triangle_type...>(base_matrix(std::forward<Arg>(arg)));
  }


  /// Make a default axes-only Covariance, based on a typed matrix template type, specifying a triangle type.
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  inline auto
  make_Covariance()
  {
    static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using B = typename MatrixTraits<Arg>::BaseMatrix;
    return make_Covariance<C, triangle_type, B>();
  }


  /// Make a default axes-only Covariance, based on a typed matrix template type.
  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  inline auto
  make_Covariance()
  {
    static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using B = typename MatrixTraits<Arg>::BaseMatrix;
    return make_Covariance<C, B>();
  }


  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  template<typename Coeffs, typename ArgType>
  struct MatrixTraits<OpenKalman::Covariance<Coeffs, ArgType>>
  {
    using BaseMatrix = ArgType;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static_assert(Coeffs::size == dimension);
    using Coefficients = Coeffs;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar; ///< Scalar type for this vector.

    template<std::size_t rows = dimension, std::size_t cols = dimension, typename S = Scalar>
    using StrictMatrix = typename MatrixTraits<BaseMatrix>::template StrictMatrix<rows, cols, S>;

    /// Make covariance from a covariance base.
    template<typename C = Coefficients, typename Arg, std::enable_if_t<is_covariance_base_v<Arg>, int> = 0>
    static auto make(Arg&& arg) noexcept
    {
      return make_Covariance<C>(std::forward<Arg>(arg));
    }

    static auto zero() { return Covariance<Coefficients, std::decay_t<BaseMatrix>>::zero(); }

    static auto identity() { return Covariance<Coefficients, std::decay_t<BaseMatrix>>::identity(); }
  };


} // OpenKalman

#endif //OPENKALMAN_COVARIANCE_H
