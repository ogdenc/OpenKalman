/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3COVARIANCEBASE_HPP
#define OPENKALMAN_EIGEN3COVARIANCEBASE_HPP

namespace Eigen
{
  template<typename CovarianceType>
  struct CovarianceCommaInitializer;
}

namespace OpenKalman::Eigen3::internal
{
  /**
   * Ultimate base of Covariance and SquareRootCovariance classes, general case.
   * No conversion is necessary if either
   * (1) Derived is not a square root and the nested matrix is self-adjoint; or
   * (2) Derived is a square root and the nested matrix is triangular.
   */
#ifdef __cpp_concepts
  template<typename Derived, typename Nested> requires
    (self_adjoint_matrix<Nested> and not square_root_covariance<Derived>) or
    (triangular_matrix<Nested> and square_root_covariance<Derived>)
  struct Eigen3CovarianceBase<Derived, Nested>
#else
  template<typename Derived, typename Nested>
  struct Eigen3CovarianceBase<Derived, Nested, std::enable_if_t<
    (self_adjoint_matrix<Nested> and not square_root_covariance<Derived>) or
    (triangular_matrix<Nested> and square_root_covariance<Derived>)>>
#endif
    : Eigen3MatrixBase<Derived, Nested> {};


  /**
   * Ultimate base of Covariance and SquareRootCovariance classes, if Derived is not a square root and
   * the nested matrix is not self-adjoint (i.e., it is triangular but not diagonal).
   */
#ifdef __cpp_concepts
  template<typename Derived, typename ArgType> requires
    (not self_adjoint_matrix<ArgType>) and (not square_root_covariance<Derived>)
  struct Eigen3CovarianceBase<Derived, ArgType>
#else
  template<typename Derived, typename ArgType>
  struct Eigen3CovarianceBase<Derived, ArgType, std::enable_if_t<
    not self_adjoint_matrix<ArgType> and not square_root_covariance<Derived>>>
#endif
    : Eigen3MatrixBase<Derived, ArgType>
  {
    using Nested = std::decay_t<ArgType>;
    using Scalar = typename Nested::Scalar;
    using Base = Eigen3MatrixBase<Derived, Nested>;

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    constexpr auto operator<<(const S& s)
    {
      auto& xpr = static_cast<Derived&>(*this);
      return Eigen::CovarianceCommaInitializer(xpr, static_cast<const Scalar&>(s));
    }

    template<typename OtherDerived>
    constexpr auto operator<<(const Eigen::DenseBase<OtherDerived>& other)
    {
      auto& xpr = static_cast<Derived&>(*this);
      return Eigen::CovarianceCommaInitializer(xpr, other);
    }
  };


  /**
   * Ultimate base of Covariance and SquareRootCovariance classes, if Derived is a square root and
   * the nested matrix is not triangular (i.e., it is self-adjoint but not diagonal).
   */
#ifdef __cpp_concepts
  template<typename Derived, typename ArgType> requires
    (not triangular_matrix<ArgType>) and square_root_covariance<Derived>
  struct Eigen3CovarianceBase<Derived, ArgType>
#else
  template<typename Derived, typename ArgType>
  struct Eigen3CovarianceBase<Derived, ArgType, std::enable_if_t<
    not triangular_matrix<ArgType> and square_root_covariance<Derived>>>
#endif
    : Eigen3MatrixBase<Derived, ArgType>
  {
    using Nested = std::decay_t<ArgType>;
    using Scalar = typename Nested::Scalar;
    using Base = Eigen3MatrixBase<Derived, Nested>;

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    constexpr auto operator<<(const S& s)
    {
      auto& xpr = static_cast<Derived&>(*this);
      return Eigen::CovarianceCommaInitializer(xpr, static_cast<const Scalar&>(s));
    }

    template<typename OtherDerived>
    constexpr auto operator<<(const Eigen::DenseBase<OtherDerived>& other)
    {
      auto& xpr = static_cast<Derived&>(*this);
      return Eigen::CovarianceCommaInitializer(xpr, other);
    }
  };

} // namespace OpenKalman::Eigen3::internal


namespace Eigen
{
  /**
   * Alternative version of CommaInitializer for Covariance and SquareRootCovariance.
   */
  template<typename CovarianceType>
  struct CovarianceCommaInitializer
  {
    using Scalar = typename CovarianceType::Scalar;
    using NestedMatrix = OpenKalman::native_matrix_t<typename OpenKalman::nested_matrix_t<CovarianceType>>;
    using Nested = CommaInitializer<NestedMatrix>;

    NestedMatrix matrix;
    Nested comma_initializer;
    CovarianceType& cov;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    CovarianceCommaInitializer(CovarianceType& xpr, const S& s)
      : matrix(), comma_initializer(matrix, static_cast<const Scalar&>(s)), cov(xpr) {}

    template<typename OtherDerived>
    CovarianceCommaInitializer(CovarianceType& xpr, const DenseBase <OtherDerived>& other)
      : matrix(), comma_initializer(matrix, other), cov(xpr) {}

    CovarianceCommaInitializer(const CovarianceCommaInitializer& o)
      : matrix(o.matrix), comma_initializer(o.comma_initializer), cov(o.cov) {}

    CovarianceCommaInitializer(CovarianceCommaInitializer&& o)
      : matrix(std::move(o.matrix)),
        comma_initializer(std::move(o.comma_initializer)), cov(std::move(o.cov)) {}

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
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

    ~CovarianceCommaInitializer()
    {
      finished();
    }

    auto& finished()
    {
      cov = comma_initializer.finished();
      return cov;
    }
  };

} // namespace Eigen


#endif //OPENKALMAN_EIGEN3COVARIANCEBASE_HPP
