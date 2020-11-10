/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3MATRIXBASE_HPP
#define OPENKALMAN_EIGEN3MATRIXBASE_HPP

namespace Eigen
{
  template<typename Derived, typename XprType>
  struct MeanCommaInitializer;

  template<typename XprType>
  struct DiagonalCommaInitializer;
} // namespace Eigen

namespace OpenKalman::Eigen3::internal
{
  /*
   * Base class for all OpenKalman classes that are also Eigen3 matrices.
   */
  template<typename Derived, typename ArgType>
  struct Eigen3MatrixBase : Eigen3Base<Derived>
  {
    using Nested = std::decay_t<ArgType>; ///< The nested Eigen matrix type. Eigen3 requires this to be defined.
    using Scalar = typename Nested::Scalar;

  protected:
    template<typename Arg>
    constexpr decltype(auto) get_ultimate_base_matrix_impl(Arg&& arg) noexcept
    {
      decltype(auto) b = base_matrix(std::forward<Arg>(arg));
      using B = decltype(b);
      if constexpr(
        Eigen3::eigen_self_adjoint_expr<B> or
        Eigen3::eigen_triangular_expr<B> or
        Eigen3::eigen_diagonal_expr<B> or
        Eigen3::euclidean_expr<B>)
      {
        return get_ultimate_base_matrix(b);
      }
      else
      {
        return b;
      }
    }

    template<typename Arg>
    constexpr decltype(auto) get_ultimate_base_matrix(Arg&& arg) noexcept
    {
      if constexpr(Eigen3::eigen_self_adjoint_expr<Arg>)
      {
        if constexpr (MatrixTraits<Arg>::storage_type == TriangleType::diagonal) return std::forward<Arg>(arg);
        else return get_ultimate_base_matrix_impl(std::forward<Arg>(arg));
      }
      else if constexpr(Eigen3::eigen_triangular_expr<Arg>)
      {
        if constexpr(MatrixTraits<Arg>::triangle_type == TriangleType::diagonal) return std::forward<Arg>(arg);
        else return get_ultimate_base_matrix_impl(std::forward<Arg>(arg));
      }
      else
      {
        return get_ultimate_base_matrix_impl(std::forward<Arg>(arg));
      }
    }

  public:

    static constexpr Eigen::Index rows() { return Eigen::internal::traits<Derived>::RowsAtCompileTime; } ///< Required by Eigen::EigenBase.

    static constexpr Eigen::Index cols() { return Eigen::internal::traits<Derived>::ColsAtCompileTime; } ///< Required by Eigen::EigenBase.

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    constexpr auto operator<<(const S& s)
    {
      auto& xpr = get_ultimate_base_matrix(static_cast<Derived&>(*this));
      using Xpr = std::decay_t<decltype(xpr)>;
      if constexpr(mean<Derived>)
      {
        return Eigen::MeanCommaInitializer<Derived, Xpr>(xpr, static_cast<const Scalar&>(s));
      }
      else if constexpr(Eigen3::eigen_self_adjoint_expr<Xpr> or Eigen3::eigen_triangular_expr<Xpr>)
      {
        return Eigen::DiagonalCommaInitializer(xpr, static_cast<const Scalar&>(s));
      }
      else
      {
        return Eigen::CommaInitializer(xpr, static_cast<const Scalar&>(s));
      }
    }

    template<typename OtherDerived>
    constexpr auto operator<<(const Eigen::DenseBase<OtherDerived>& other)
    {
      auto& xpr = get_ultimate_base_matrix(static_cast<Derived&>(*this));
      using Xpr = std::decay_t<decltype(xpr)>;
      if constexpr(mean<Derived>)
      {
        return Eigen::MeanCommaInitializer<Derived, Xpr>(xpr, static_cast<const OtherDerived&>(other));
      }
      else if constexpr(Eigen3::eigen_self_adjoint_expr<Xpr> or Eigen3::eigen_triangular_expr<Xpr>)
      {
        std::cout << "a3" << std::endl << std::flush;
        return Eigen::DiagonalCommaInitializer(xpr, static_cast<const OtherDerived&>(other));
      }
      else
      {
        std::cout << "a4" << std::endl << std::flush;
        return Eigen::CommaInitializer(xpr, static_cast<const OtherDerived&>(other));
      }
    }

    /// Refined to avoid confusion with zero().
    static decltype(auto) Zero() { return Derived::zero(); }

    /// Redefined to avoid confusion with identity().
    static decltype(auto) Identity() { return Derived::identity(); }

  };

} // namespace OpenKalman::Eigen3::internal

namespace Eigen
{
  /**
   * Alternative version of CommaInitializer for Mean.
   */
  template<typename Derived, typename XprType>
  struct MeanCommaInitializer : CommaInitializer<XprType>
  {
    using Base = CommaInitializer<XprType>;
    using Scalar = typename XprType::Scalar;
    using Coefficients = typename OpenKalman::MatrixTraits<Derived>::RowCoefficients;
    using Base::Base;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    MeanCommaInitializer(XprType& xpr, const S& s) : Base(xpr, static_cast<const Scalar&>(s)) {}

    template<typename OtherDerived>
    MeanCommaInitializer(XprType& xpr, const DenseBase <OtherDerived>& other) : Base(xpr, other) {}

    ~MeanCommaInitializer()
    {
      finished();
    }

    auto& finished()
    {
      this->m_xpr = OpenKalman::Eigen3::wrap_angles<Coefficients>(Base::finished());
      return this->m_xpr;
    }
  };

  /**
   * Alternative version of CommaInitializer for diagonal versions of SelfAdjointMatrix and TriangularMatrix.
   */
  template<typename XprType>
  struct DiagonalCommaInitializer
  {
    using Scalar = typename XprType::Scalar;
    static constexpr auto dim = OpenKalman::MatrixTraits<XprType>::dimension;
    using BaseMatrix = OpenKalman::native_matrix_t<typename OpenKalman::MatrixTraits<XprType>::BaseMatrix, dim, 1>;
    using Nested = CommaInitializer<BaseMatrix>;

    BaseMatrix matrix;
    Nested comma_initializer;
    XprType& diag;

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    DiagonalCommaInitializer(XprType& xpr, const S& s)
      : matrix(), comma_initializer(matrix, static_cast<const Scalar&>(s)), diag(xpr) {}

    template<typename OtherDerived>
    DiagonalCommaInitializer(XprType& xpr, const DenseBase <OtherDerived>& other)
      : matrix(), comma_initializer(matrix, other), diag(xpr) {}

    DiagonalCommaInitializer(const DiagonalCommaInitializer& o)
      : matrix(o.matrix), comma_initializer(o.comma_initializer), diag(o.diag) {}

    DiagonalCommaInitializer(DiagonalCommaInitializer&& o)
      : matrix(std::move(o.matrix)),
        comma_initializer(std::move(o.comma_initializer)), diag(std::move(o.diag)) {}

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator,(const S& s)
    {
      comma_initializer, static_cast<const Scalar&>(s);
      return *this;
    }

    template<typename OtherDerived>
    auto& operator,(const DenseBase<OtherDerived>& other)
    {
      comma_initializer, other;
      return *this;
    }

    ~DiagonalCommaInitializer()
    {
      finished();
    }

    auto& finished()
    {
      diag = OpenKalman::Eigen3::DiagonalMatrix<BaseMatrix>(comma_initializer.finished());
      return diag;
    }
  };

} // namespace Eigen3

#endif //OPENKALMAN_EIGEN3MATRIXBASE_HPP
