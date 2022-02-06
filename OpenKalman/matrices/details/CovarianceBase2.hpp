/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for CovarianceBase, case 2.
 */

#ifndef OPENKALMAN_COVARIANCEBASE2_HPP
#define OPENKALMAN_COVARIANCEBASE2_HPP

namespace OpenKalman::internal
{
  // ======================================CASE 2======================================
  /**
   * \internal
   * \anchor CovarianceBaseCase2
   * \brief Base of Covariance and SquareRootCovariance classes (Case 2).
   * \details This specialization is operative if NestedMatrix is not self-contained and either
   * # Derived is not a square root and NestedMatrix is self-adjoint; or
   * # Derived is a square root and NestedMatrix is triangular.
   * In this case, NestedMatrix and the cholesky nested matrix are the same.
   */
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix> requires
    case1or2<Derived, NestedMatrix> and (not self_contained<NestedMatrix>)
  struct CovarianceBase<Derived, NestedMatrix>
#else
  template<typename Derived, typename NestedMatrix>
  struct CovarianceBase<Derived, NestedMatrix, std::enable_if_t<
    case1or2<Derived, NestedMatrix> and (not self_contained<NestedMatrix>)>>
#endif
    : MatrixBase<Derived, NestedMatrix>
  {
  private:

    using Base = MatrixBase<Derived, NestedMatrix>;

    using Scalar = scalar_type_of_t<NestedMatrix>;


#ifdef __cpp_concepts
    template<typename, typename>
#else
    template<typename, typename, typename>
#endif
    friend struct CovarianceBase;

  protected:

    using CholeskyNestedMatrix = std::conditional_t<
      diagonal_matrix<NestedMatrix>,
      typename MatrixTraits<NestedMatrix>::template DiagonalMatrixFrom<>,
      std::conditional_t<triangular_matrix<NestedMatrix>,
        typename MatrixTraits<NestedMatrix>::template SelfAdjointMatrixFrom<>,
        typename MatrixTraits<NestedMatrix>::template TriangularMatrixFrom<>>>;


    /**
     * \internal
     * \brief The Cholesky nested matrix: triangular if Covariance, self-adjoint if SquareRootCovariance
     */
    const std::function<CholeskyNestedMatrix()> cholesky_nested_matrix;


    /**
     * \internal
     * \brief The synchronization direction. 0 == synchronized, 1 = forward, -1 = reverse.
     */
    const std::function<int()> synchronization_direction;


    /**
     * \internal
     * \brief Synchronize the state from nested_matrix to cholesky_nested_matrix.
     * \details This is a no-op.
     */
    const std::function<void()> synchronize_forward;


    /**
     * \internal
     * \brief Synchronize the state from cholesky_nested_matrix to nested_matrix.
     * \details This is a no-op.
     */
    const std::function<void()> synchronize_reverse;


    /**
     * \internal
     * \brief Indicate that the nested matrix has changed.
     */
    const std::function<void()> mark_nested_matrix_changed;


    /**
     * \internal
     * \brief Indicate that the cholesky nested matrix has changed.
     * \details In this specialization of CovarianceBase, this function is a no-op.
     */
    constexpr static auto mark_cholesky_nested_matrix_changed = [] {};


    /**
     * \internal
     * \brief Indicate that the covariance is synchronized.
     * \details In this specialization of CovarianceBase, this function is a no-op.
     */
    constexpr static auto mark_synchronized = [] {};


  public:
    /// Default constructor.
    CovarianceBase() = delete;


    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other)
      : Base {other},
        cholesky_nested_matrix {other.cholesky_nested_matrix},
        synchronization_direction {other.synchronization_direction},
        synchronize_forward {other.synchronize_forward},
        synchronize_reverse {other.synchronize_reverse},
        mark_nested_matrix_changed {other.mark_nested_matrix_changed}
        {}


    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : Base {std::move(other)},
        cholesky_nested_matrix {std::move(other).cholesky_nested_matrix},
        synchronization_direction {std::move(other).synchronization_direction},
        synchronize_forward {std::move(other).synchronize_forward},
        synchronize_reverse {std::move(other).synchronize_reverse},
        mark_nested_matrix_changed {std::move(other).mark_nested_matrix_changed}
        {}


    /**
     * \brief Construct from an lvalue reference to a "Case 1" \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires case1or2<Arg> and self_contained<nested_matrix_of<Arg>>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and case1or2<Arg> and
      self_contained<nested_matrix_of<Arg>>, int> = 0>
#endif
    CovarianceBase(Arg& arg)
      : Base {arg.nested_matrix()},
        cholesky_nested_matrix {[&arg] { return arg.cholesky_nested_matrix(); }},
        synchronization_direction {[]() -> int { return 0; }},
        synchronize_forward {[] {}},
        synchronize_reverse {[] {}},
        mark_nested_matrix_changed {[] {}}
        {}


    /**
     * \brief Construct from another "Case 2" \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires case1or2<Arg> and (not self_contained<nested_matrix_of<Arg>>) and
      (not std::derived_from<std::decay_t<Arg>, CovarianceBase>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and case1or2<Arg> and
      (not self_contained<nested_matrix_of<Arg>>) and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>), int> = 0>
#endif
    CovarianceBase(Arg&& arg)
      : Base {std::forward<Arg>(arg).nested_matrix()},
        cholesky_nested_matrix {std::forward<Arg>(arg).cholesky_nested_matrix},
        synchronization_direction {std::forward<Arg>(arg).synchronization_direction},
        synchronize_forward {std::forward<Arg>(arg).synchronize_forward},
        synchronize_reverse {std::forward<Arg>(arg).synchronize_reverse},
        mark_nested_matrix_changed {std::forward<Arg>(arg).mark_nested_matrix_changed}
    {}


    /**
     * \brief Construct from an lvalue reference to a "Case 3" \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not case1or2<Arg>) and self_contained<nested_matrix_of<Arg>>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and (not case1or2<Arg>) and
      self_contained<nested_matrix_of<Arg>>, int> = 0>
#endif
    CovarianceBase(Arg& arg)
      : Base {arg.nested_matrix()},
        cholesky_nested_matrix {[&arg] { return arg.cholesky_nested; }},
        synchronization_direction {[&arg] { return arg.synch_direction; }},
        synchronize_forward {[&arg] { arg.synchronize_forward(); }},
        synchronize_reverse {[&arg] { arg.synchronize_reverse(); }},
        mark_nested_matrix_changed {[&arg] { arg.mark_nested_matrix_changed(); }}
    {}


    /**
     * \brief Construct from an lvalue reference to "Case 4" \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not case1or2<Arg>) and (not self_contained<nested_matrix_of<Arg>>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not case1or2<Arg>) and (not self_contained<nested_matrix_of<Arg>>), int> = 0>
#endif
    CovarianceBase(Arg& arg)
      : Base {arg.nested_matrix()},
        cholesky_nested_matrix {[&arg] { return arg.cholesky_nested_link; }},
        synchronization_direction {arg.synchronization_direction},
        synchronize_forward {arg.synchronize_forward},
        synchronize_reverse {arg.synchronize_reverse},
        mark_nested_matrix_changed {arg.mark_nested_matrix_changed}
    {}


    /**
     * \brief Construct from an rvalue reference to a "Case 4" \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not case1or2<Arg>) and (not self_contained<nested_matrix_of<Arg>>) and
      std::is_rvalue_reference_v<Arg&&>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and (not case1or2<Arg>) and
      (not self_contained<nested_matrix_of<Arg>>) and std::is_rvalue_reference_v<Arg&&>, int> = 0>
#endif
    CovarianceBase(Arg&& arg)
      : Base {std::move(arg).nested_matrix()},
        cholesky_nested_matrix {[this] { return to_covariance_nestable<CholeskyNestedMatrix>(Base::nested_matrix()); }},
        synchronization_direction {[]() -> int { return 0; }},
        synchronize_forward {[] {}},
        synchronize_reverse {[] {}},
        mark_nested_matrix_changed {[] {}}
    {}


    /**
     * \brief Construct from a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
    explicit CovarianceBase(Arg&& arg) noexcept
      : Base {std::forward<Arg>(arg)},
        cholesky_nested_matrix {[this] {
          return to_covariance_nestable<CholeskyNestedMatrix>(Base::nested_matrix());
        }},
        synchronization_direction {[]() -> int { return 0; }},
        synchronize_forward {[] {}},
        synchronize_reverse {[] {}},
        mark_nested_matrix_changed {[] {}}
        {}


    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        Base::operator=(other);
        mark_nested_matrix_changed();
      }
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        Base::operator=(std::move(other));
        mark_nested_matrix_changed();
      }
      return *this;
    }


    /**
     * \internal
     * \brief Assign from another \ref covariance or a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (covariance<Arg> or covariance_nestable<Arg>) and
      (not std::derived_from<std::decay_t<Arg>, CovarianceBase>)
#else
    template<typename Arg, std::enable_if_t<(covariance<Arg> or covariance_nestable<Arg>) and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>), int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      Base::operator=(to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg)));
      mark_nested_matrix_changed();
      return *this;
    }


    /**
     * \brief Get or set element (i, j) of the covariance matrix.
     * \param i The row.
     * \param j The column.
     * \return An ElementAccessor object.
     */
    auto operator() (std::size_t i, std::size_t j)
    {
      if constexpr (element_settable<NestedMatrix, std::size_t, std::size_t>)
        return ElementAccessor(Base::nested_matrix(), i, j,
          [this] { if (synchronization_direction() < 0) synchronize_reverse(); },
          [this] { mark_nested_matrix_changed(); });
      else
        return ElementAccessor(Base::nested_matrix(), i, j);
    }


    /// \overload
    auto operator() (std::size_t i, std::size_t j) const
    {
      return ElementAccessor(Base::nested_matrix(), i, j);
    }


    /**
     * \brief Get or set element i of the covariance matrix, if it is a vector.
     * \param i The row.
     * \return An ElementAccessor object.
     */
    auto operator[] (std::size_t i)
    {
      if constexpr (element_settable<NestedMatrix, std::size_t>)
        return ElementAccessor(Base::nested_matrix(), i,
          [this] { if (synchronization_direction() < 0) synchronize_reverse(); },
          [this] { mark_nested_matrix_changed(); });
      else
        return ElementAccessor(Base::nested_matrix(), i);
    }


    /// \overload
    auto operator[] (std::size_t i) const
    {
      return ElementAccessor(Base::nested_matrix(), i);
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i, const std::size_t j)
    {
      OpenKalman::set_element(Base::nested_matrix(), s, i, j);
      mark_nested_matrix_changed();
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i)
    {
      OpenKalman::set_element(Base::nested_matrix(), s, i);
      mark_nested_matrix_changed();
    }

  };


}

#endif //OPENKALMAN_COVARIANCEBASE2_HPP
